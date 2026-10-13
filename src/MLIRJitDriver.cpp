#include "mlir_psp/MLIRJitDriver.h"
#include "mlir_psp/Profiling.h"
#ifdef MLIR_PSP_ENABLE_PROFILING
#include "llvm/IR/PassTimingInfo.h"
#endif

#include <unordered_map>

#include "mlir/IR/MLIRContext.h"

namespace MLIRPSP {

namespace {

struct CachedBlock {
	uint32_t guestAddress = 0;
	uint32_t mipsBytes = 0;
	std::unique_ptr<CompiledMLIRBlock> executable;
};

bool RangesOverlap(uint32_t firstAddress, uint32_t firstLength,
		uint32_t secondAddress, uint32_t secondLength) {
	const uint64_t firstEnd = static_cast<uint64_t>(firstAddress) + firstLength;
	const uint64_t secondEnd = static_cast<uint64_t>(secondAddress) + secondLength;
	return static_cast<uint64_t>(firstAddress) < secondEnd &&
		static_cast<uint64_t>(secondAddress) < firstEnd;
}

}  // namespace

struct MLIRJitDriver::Impl {
	mlir::MLIRContext context;
	std::unordered_map<uint32_t, CachedBlock> blocks;
};

MLIRJitDriver::MLIRJitDriver() : impl_(std::make_unique<Impl>()) {}
MLIRJitDriver::~MLIRJitDriver() {
#ifdef MLIR_PSP_ENABLE_PROFILING
	llvm::reportAndResetTimings();
#endif
}
MLIRJitDriver::MLIRJitDriver(MLIRJitDriver &&) noexcept = default;
MLIRJitDriver &MLIRJitDriver::operator=(MLIRJitDriver &&) noexcept = default;

DriverPrepareResult MLIRJitDriver::PrepareBlock(uint32_t guestAddress,
		uint32_t mipsBytes, const std::vector<IRInst> &instructions) {
	DriverPrepareResult result;
	if (impl_->blocks.find(guestAddress) != impl_->blocks.end()) {
		result.compileStatus = CompileStatus::Compiled;
		result.cacheHit = true;
		return result;
	}

	auto compiled = [&] {
		MLIR_PSP_PROFILE(Lower);
		return CompileBlockToMLIR(impl_->context, instructions, guestAddress);
	}();
	result.compileStatus = compiled.status;
	if (!compiled) {
		result.error = std::move(compiled.error);
		return result;
	}
	auto executable = [&] {
		MLIR_PSP_PROFILE(Compile);
		return CompileToExecutable(*compiled.module, compiled.functionName);
	}();
	if (!executable) {
		result.compileStatus = CompileStatus::Error;
		result.error = std::move(executable.error);
		return result;
	}
	CachedBlock cached{guestAddress, mipsBytes, std::move(executable.block)};
	impl_->blocks.emplace(guestAddress, std::move(cached));
	result.compileStatus = CompileStatus::Compiled;
	return result;
}

DriverExecutionResult MLIRJitDriver::ExecuteBlock(uint32_t guestAddress,
		uint32_t mipsBytes, const std::vector<IRInst> &instructions,
		uint32_t *stateWords, size_t stateWordCount, uint32_t *memoryWords,
		size_t memoryWordCount, const MemoryAccessCallbacks *memoryCallbacks) {
	auto prepared = PrepareBlock(guestAddress, mipsBytes, instructions);
	if (!prepared) {
		DriverExecutionResult result;
		result.compileStatus = prepared.compileStatus;
		result.cacheHit = prepared.cacheHit;
		result.error = std::move(prepared.error);
		return result;
	}
	auto result = ExecuteCachedBlock(guestAddress, stateWords, stateWordCount,
		memoryWords, memoryWordCount, memoryCallbacks);
	// A block compiled by this call was not an execution-cache hit.
	result.cacheHit = prepared.cacheHit;
	return result;
}

DriverExecutionResult MLIRJitDriver::ExecuteCachedBlock(uint32_t guestAddress,
		uint32_t *stateWords, size_t stateWordCount, uint32_t *memoryWords,
		size_t memoryWordCount, const MemoryAccessCallbacks *memoryCallbacks) {
	DriverExecutionResult result;
	result.compileStatus = CompileStatus::Compiled;
	MLIR_PSP_PROFILE_HOT(Dispatch);
	result.cacheHit = true;
	auto found = impl_->blocks.find(guestAddress);
	if (found == impl_->blocks.end()) {
		result.compileStatus = CompileStatus::Error;
		result.cacheHit = false;
		result.error = "requested MLIR block is not cached";
		return result;
	}
	result.execution = found->second.executable->Execute(stateWords, stateWordCount,
		memoryWords, memoryWordCount, memoryCallbacks);
	if (!result.execution)
		result.error = result.execution.error;
	return result;
}

void MLIRJitDriver::ClearCache() {
	impl_->blocks.clear();
}

void MLIRJitDriver::InvalidateCache(uint32_t guestAddress, uint32_t length) {
	if (length == 0)
		return;
	for (auto it = impl_->blocks.begin(); it != impl_->blocks.end();) {
		const auto &block = it->second;
		if (RangesOverlap(block.guestAddress, block.mipsBytes, guestAddress, length))
			it = impl_->blocks.erase(it);
		else
			++it;
	}
}

bool MLIRJitDriver::HasCachedBlock(uint32_t guestAddress) const {
	return impl_->blocks.find(guestAddress) != impl_->blocks.end();
}

size_t MLIRJitDriver::CachedBlockCount() const {
	return impl_->blocks.size();
}

}  // namespace MLIRPSP
