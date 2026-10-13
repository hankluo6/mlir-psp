#include "mlir_psp/MLIRJitEngine.h"
#include "mlir_psp/Profiling.h"
#ifdef MLIR_PSP_ENABLE_PROFILING
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include <cstdlib>
#endif
#include "mlir_psp/MIPSStateLayout.h"

#include <cmath>
#include <cstring>
#include <mutex>

#ifdef MLIR_PSP_ADAPTER_ENABLE_JIT
#include "Core/MIPS/MIPSVFPUUtils.h"
#endif

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#include <immintrin.h>
#endif

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"

namespace MLIRPSP {
namespace {

struct ActiveMemoryContext {
	uint32_t *words = nullptr;
	size_t wordCount = 0;
	const MemoryAccessCallbacks *callbacks = nullptr;
	MemoryAccessStatus status = MemoryAccessStatus::Success;
};

thread_local ActiveMemoryContext *activeMemory = nullptr;
thread_local uint32_t *activeState = nullptr;

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
thread_local uint32_t hostMxcsr = 0;

void ApplyGuestRounding(uint32_t fcr31) {
	uint32_t mode = fcr31 & 3;
	if (mode & 1)
		mode ^= 2;
	uint32_t csr = _mm_getcsr() & ~(7u << 13);
	csr |= mode << 13;
	if (fcr31 & (1u << 24))
		csr |= 1u << 15;
	_mm_setcsr(csr);
}
#endif

void UpdateFcr31Exceptions(uint32_t hostExceptions) {
	if (!activeState)
		return;
	uint32_t cause = 0;
	uint32_t flags = 0;
	auto addException = [&](uint32_t hostBit, uint32_t causeBit, uint32_t flagBit) {
		if (hostExceptions & hostBit) {
			cause |= causeBit;
			flags |= flagBit;
		}
	};
	addException(1u << 0, 1u << 16, 1u << 6);  // Invalid operation.
	addException(1u << 2, 1u << 15, 1u << 5);  // Division by zero.
	addException(1u << 3, 1u << 14, 1u << 4);  // Overflow.
	addException(1u << 4, 1u << 13, 1u << 3);  // Underflow.
	addException(1u << 5, 1u << 12, 1u << 2);  // Inexact.
	uint32_t &fcr31 = activeState[MIPSStateLayout::FCR31];
	fcr31 = (fcr31 & ~0x0001F000u) | cause | flags;
}

extern "C" void mlir_psp_apply_rounding(uint32_t fcr31) {
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	ApplyGuestRounding(fcr31);
#else
	(void)fcr31;
#endif
}

extern "C" void mlir_psp_restore_rounding() {
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	_mm_setcsr(hostMxcsr);
#endif
}

extern "C" uint32_t mlir_psp_memory_read(uint32_t address, uint32_t size,
		uint32_t signExtend) {
	MLIR_PSP_PROFILE_HOT(Read);
	if (!activeMemory)
		return 0;
	if (activeMemory->callbacks && activeMemory->callbacks->read) {
		auto result = activeMemory->callbacks->read(
			activeMemory->callbacks->userdata, address, size, signExtend != 0);
		if (activeMemory->status == MemoryAccessStatus::Success)
			activeMemory->status = result.status;
		return result.value;
	}
	size_t byteOffset = address & 0x01FFFFFFu;
	if ((size != 1 && size != 2 && size != 4) ||
			byteOffset + size > activeMemory->wordCount * sizeof(uint32_t)) {
		activeMemory->status = MemoryAccessStatus::InvalidAddress;
		return 0;
	}
	uint32_t value = 0;
	std::memcpy(&value, reinterpret_cast<uint8_t *>(activeMemory->words) + byteOffset,
		size);
	if (signExtend && size == 1)
		value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(value)));
	if (signExtend && size == 2)
		value = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(value)));
	return value;
}

extern "C" void mlir_psp_memory_write(uint32_t address, uint32_t size,
		uint32_t value) {
	MLIR_PSP_PROFILE_HOT(Write);
	if (!activeMemory)
		return;
	if (activeMemory->callbacks && activeMemory->callbacks->write) {
		auto status = activeMemory->callbacks->write(
			activeMemory->callbacks->userdata, address, size, value);
		if (activeMemory->status == MemoryAccessStatus::Success)
			activeMemory->status = status;
		return;
	}
	size_t byteOffset = address & 0x01FFFFFFu;
	if ((size != 1 && size != 2 && size != 4) ||
			byteOffset + size > activeMemory->wordCount * sizeof(uint32_t)) {
		activeMemory->status = MemoryAccessStatus::InvalidAddress;
		return;
	}
	std::memcpy(reinterpret_cast<uint8_t *>(activeMemory->words) + byteOffset,
		&value, size);
}

extern "C" uint32_t mlir_psp_memory_write_conditional(uint32_t address,
		uint32_t size, uint32_t value, uint32_t condition) {
	if (!condition)
		return 0;
	mlir_psp_memory_write(address, size, value);
	return activeMemory && activeMemory->status == MemoryAccessStatus::Success ? 1 : 0;
}

extern "C" uint32_t mlir_psp_float_sqrt(uint32_t inputBits) {
	float input;
	std::memcpy(&input, &inputBits, sizeof(input));
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	_mm_setcsr(_mm_getcsr() & ~0x3Fu);
#endif
	volatile float output = std::sqrt(input);
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	UpdateFcr31Exceptions(_mm_getcsr() & 0x3Fu);
#endif
	uint32_t outputBits;
	float storedOutput = output;
	std::memcpy(&outputBits, &storedOutput, sizeof(outputBits));
	return outputBits;
}

extern "C" uint32_t mlir_psp_float_binary(uint32_t lhsBits, uint32_t rhsBits,
		uint32_t operation) {
	float lhs;
	float rhs;
	std::memcpy(&lhs, &lhsBits, sizeof(lhs));
	std::memcpy(&rhs, &rhsBits, sizeof(rhs));
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	_mm_setcsr(_mm_getcsr() & ~0x3Fu);
#endif
	volatile float output;
	switch (operation) {
	case 0: output = lhs + rhs; break;
	case 1: output = lhs - rhs; break;
	case 2: output = lhs * rhs; break;
	default: output = lhs / rhs; break;
	}
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	uint32_t exceptions = _mm_getcsr() & 0x3Fu;
	// The PSP reports invalid operation for arithmetic involving a NaN,
	// including a quiet NaN, while x86 normally propagates a quiet NaN.
	if ((lhsBits & 0x7FFFFFFFu) > 0x7F800000u ||
			(rhsBits & 0x7FFFFFFFu) > 0x7F800000u)
		exceptions |= 1u;
	UpdateFcr31Exceptions(exceptions);
#endif
	float storedOutput = output;
	uint32_t outputBits;
	std::memcpy(&outputBits, &storedOutput, sizeof(outputBits));
	// PSP scalar/VFPU arithmetic canonicalizes NaN results to a positive quiet
	// NaN.  Host arithmetic may preserve the sign bit of an input NaN.
	if ((outputBits & 0x7FFFFFFFu) > 0x7F800000u)
		outputBits = 0x7FC00000u;
	return outputBits;
}

extern "C" uint32_t mlir_psp_float_to_int(uint32_t inputBits, uint32_t mode) {
	float input;
	std::memcpy(&input, &inputBits, sizeof(input));
	const uint32_t magnitude = inputBits & 0x7FFFFFFFu;
	if (magnitude >= 0x7F800000u)
		return magnitude == 0x7F800000u && (inputBits >> 31) != 0
			? 0x80000000u : 0x7FFFFFFFu;

	double rounded;
	switch (mode & 3) {
	case 0: {
		double integral = std::floor(input);
		double fraction = static_cast<double>(input) - integral;
		if (fraction > 0.5 || (fraction == 0.5 &&
				std::fmod(integral, 2.0) != 0.0))
			integral += 1.0;
		rounded = integral;
		break;
	}
	case 1:
		rounded = input >= 0.0f ? std::floor(input) : std::ceil(input);
		break;
	case 2:
		rounded = std::ceil(input);
		break;
	default:
		rounded = std::floor(input);
		break;
	}
	int32_t result = static_cast<int32_t>(rounded);
	if ((mode & 3) == 1 && input >= 0.0f && result == INT32_MIN)
		result = INT32_MAX;
	return static_cast<uint32_t>(result);
}

extern "C" uint32_t mlir_psp_float_to_scaled_int(uint32_t inputBits,
		uint32_t scale, uint32_t mode) {
	float input;
	std::memcpy(&input, &inputBits, sizeof(input));
	if (std::isnan(input))
		return 0x7FFFFFFFu;
	double scaled = static_cast<double>(input) *
		static_cast<double>(uint64_t{1} << (scale & 31));
	if (scaled > static_cast<double>(INT32_MAX))
		return 0x7FFFFFFFu;
	if (scaled <= static_cast<double>(INT32_MIN))
		return 0x80000000u;
	double rounded;
	switch (mode & 3) {
	case 0: {
		double integral = std::floor(scaled);
		double fraction = scaled - integral;
		if (fraction > 0.5 || (fraction == 0.5 &&
			std::fmod(integral, 2.0) != 0.0))
			integral += 1.0;
		rounded = integral;
		break;
	}
	case 1: rounded = input >= 0.0f ? std::floor(scaled) : std::ceil(scaled); break;
	case 2: rounded = std::ceil(scaled); break;
	default: rounded = std::floor(scaled); break;
	}
	return static_cast<uint32_t>(static_cast<int32_t>(rounded));
}

extern "C" uint32_t mlir_psp_float_unary(uint32_t inputBits, uint32_t operation) {
	float input;
	std::memcpy(&input, &inputBits, sizeof(input));
	constexpr float halfPi = 1.57079632679489661923f;
	float output;
	switch (operation) {
	case 0: output = std::sin(halfPi * input); break;
	case 1: output = std::cos(halfPi * input); break;
	case 2: output = 1.0f / std::sqrt(input); break;
	case 3: output = 1.0f / input; break;
	default:
#ifdef MLIR_PSP_ADAPTER_ENABLE_JIT
		// Use PPSSPP's PSP-accurate tables when running inside its runtime.
		output = vfpu_asin(input);
#else
		// Match PPSSPP's no-table fallback: float asin, double division.
		output = static_cast<float>(std::asin(input) / 1.57079632679489661923);
#endif
		break;
	}
	uint32_t outputBits;
	std::memcpy(&outputBits, &output, sizeof(outputBits));
	return outputBits;
}

std::string ErrorToString(llvm::Error error) {
	return llvm::toString(std::move(error));
}

}  // namespace

struct CompiledMLIRBlock::Impl {
	using PackedEntryPoint = void (*)(void **);

	std::unique_ptr<mlir::ExecutionEngine> engine;
	PackedEntryPoint entryPoint = nullptr;
};

CompiledMLIRBlock::CompiledMLIRBlock() = default;
CompiledMLIRBlock::CompiledMLIRBlock(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CompiledMLIRBlock::~CompiledMLIRBlock() = default;
CompiledMLIRBlock::CompiledMLIRBlock(CompiledMLIRBlock &&) noexcept = default;
CompiledMLIRBlock &CompiledMLIRBlock::operator=(CompiledMLIRBlock &&) noexcept = default;

CompileExecutableResult CompileToExecutable(mlir::ModuleOp module,
		const std::string &functionName) {
	CompileExecutableResult result;

	mlir::registerBuiltinDialectTranslation(*module.getContext());
	mlir::registerLLVMDialectTranslation(*module.getContext());

	mlir::PassManager passManager(module.getContext());
	passManager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
	passManager.addNestedPass<mlir::func::FuncOp>(mlir::createArithToLLVMConversionPass());
	passManager.addPass(mlir::createConvertControlFlowToLLVMPass());
	passManager.addPass(mlir::createConvertFuncToLLVMPass());
	passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
	auto passResult = [&] {
		MLIR_PSP_PROFILE(Passes);
		return passManager.run(module);
	}();
	if (mlir::failed(passResult)) {
		result.error = "MLIR-to-LLVM lowering failed";
		return result;
	}

	static std::once_flag initializeTarget;
	std::call_once(initializeTarget, [] {
#ifdef MLIR_PSP_ENABLE_PROFILING
		// Diagnostic only: LLVM's process-wide legacy codegen pass timers.
		if (const char *timing = std::getenv("MLIR_PSP_TIME_LLVM_PASSES"))
			if (std::strcmp(timing, "1") == 0)
				llvm::TimePassesIsEnabled = true;
#endif
		llvm::InitializeNativeTarget();
		llvm::InitializeNativeTargetAsmPrinter();
	});
	auto engineOrError = [&] {
		MLIR_PSP_PROFILE(EngineCreate);
#ifdef MLIR_PSP_ENABLE_PROFILING
		if (const char *level = std::getenv("MLIR_PSP_CODEGEN_LEVEL")) {
			if (level[0] < '0' || level[0] > '3' || level[1] != '\0')
				return llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>>(
					llvm::createStringError(llvm::inconvertibleErrorCode(),
						"MLIR_PSP_CODEGEN_LEVEL must be 0, 1, 2 or 3"));
			auto builder = llvm::orc::JITTargetMachineBuilder::detectHost();
			if (!builder)
				return llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>>(builder.takeError());
			builder->setCodeGenOptLevel(static_cast<llvm::CodeGenOptLevel>(level[0] - '0'));
			auto machine = builder->createTargetMachine();
			if (!machine)
				return llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>>(machine.takeError());
			return mlir::ExecutionEngine::create(module, {}, std::move(*machine));
		}
#endif
		return mlir::ExecutionEngine::create(module);
	}();
	if (!engineOrError) {
		result.error = ErrorToString(engineOrError.takeError());
		return result;
	}
	auto impl = std::make_unique<CompiledMLIRBlock::Impl>();
	impl->engine = std::move(*engineOrError);
	impl->engine->registerSymbols([](llvm::orc::MangleAndInterner interner) {
		llvm::orc::SymbolMap symbols;
		symbols[interner("mlir_psp_memory_read")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_memory_read),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_memory_write")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_memory_write),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_memory_write_conditional")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_memory_write_conditional),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_float_sqrt")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_float_sqrt),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_float_binary")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_float_binary),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_float_to_int")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_float_to_int),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_float_to_scaled_int")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_float_to_scaled_int),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_float_unary")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_float_unary),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_apply_rounding")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_apply_rounding),
			llvm::JITSymbolFlags::Exported};
		symbols[interner("mlir_psp_restore_rounding")] = {
			llvm::orc::ExecutorAddr::fromPtr(&mlir_psp_restore_rounding),
			llvm::JITSymbolFlags::Exported};
		return symbols;
	});
	{
		MLIR_PSP_PROFILE(Initialize);
		impl->engine->initialize();
	}
	const std::string cInterfaceName = "_mlir_ciface_" + functionName;
	auto entryPointOrError = [&] {
		MLIR_PSP_PROFILE(Lookup);
		return impl->engine->lookupPacked(cInterfaceName);
	}();
	if (!entryPointOrError) {
		result.error = ErrorToString(entryPointOrError.takeError());
		return result;
	}
	impl->entryPoint = *entryPointOrError;
	result.block = std::unique_ptr<CompiledMLIRBlock>(
		new CompiledMLIRBlock(std::move(impl)));
	return result;
}

ExecuteResult CompiledMLIRBlock::Execute(uint32_t *stateWords, size_t stateWordCount,
		uint32_t *memoryWords, size_t memoryWordCount,
		const MemoryAccessCallbacks *memoryCallbacks) {
	ExecuteResult result;
	MLIR_PSP_PROFILE_HOT(Execute);
	if (!impl_) {
		result.error = "compiled MLIR block is empty";
		return result;
	}
	if (!stateWords || stateWordCount != MIPSStateLayout::StateWords) {
		result.error = "JIT execution requires the MLIR MIPS state ABI size";
		return result;
	}

	StridedMemRefType<uint32_t, 1> state{
		stateWords, stateWords, 0, {static_cast<int64_t>(stateWordCount)}, {1}};
	uint32_t nextPC = 0;
	uint32_t unusedMemory = 0;
	if (!memoryWords) {
		memoryWords = &unusedMemory;
		memoryWordCount = 1;
	}
	StridedMemRefType<uint32_t, 1> memory{
		memoryWords, memoryWords, 0, {static_cast<int64_t>(memoryWordCount)}, {1}};
	ActiveMemoryContext memoryContext{memoryWords, memoryWordCount,
		memoryCallbacks, MemoryAccessStatus::Success};
	activeMemory = &memoryContext;
	activeState = stateWords;
	auto *stateDescriptor = &state;
	auto *memoryDescriptor = &memory;
	void *arguments[] = {&stateDescriptor, &memoryDescriptor, &nextPC};
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	hostMxcsr = _mm_getcsr();
	ApplyGuestRounding(stateWords[MIPSStateLayout::FCR31]);
#endif
	{
		MLIR_PSP_PROFILE_HOT(Entry);
		impl_->entryPoint(arguments);
	}
#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	_mm_setcsr(hostMxcsr);
#endif
	activeMemory = nullptr;
	activeState = nullptr;
	result.nextPC = nextPC;
	result.memoryStatus = memoryContext.status;
	return result;
}

}  // namespace MLIRPSP
