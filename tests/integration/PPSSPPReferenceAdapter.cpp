#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "Common/System/Request.h"
#include "Common/System/System.h"
#include "Core/Config.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/Debugger/SymbolMap.h"
#include "Core/MemMap.h"
#include "Core/MIPS/MIPS.h"
#include "Core/MIPS/MIPSVFPUUtils.h"
#include "Core/MIPS/IR/IRFrontend.h"
#include "Core/MIPS/IR/IRInst.h"
#include "Core/System.h"
#include "mlir_psp/MIPSStateLayout.h"
#include "mlir_psp/PPSSPPMemoryAccess.h"

#ifdef MLIR_PSP_ADAPTER_ENABLE_JIT
#include "mlir_psp/PPSSPPMLIRRuntime.h"
#endif

static_assert(offsetof(MIPSState, r) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::GPRBase);
static_assert(offsetof(MIPSState, f) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::FPRBase);
static_assert(offsetof(MIPSState, v) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::VFPUBase);
static_assert(offsetof(MIPSState, pc) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::PC);
static_assert(offsetof(MIPSState, lo) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::Lo);
static_assert(offsetof(MIPSState, hi) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::Hi);
static_assert(offsetof(MIPSState, nextPC) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::NextPC);
static_assert(offsetof(MIPSState, downcount) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::Downcount);
static_assert(offsetof(MIPSState, llBit) / sizeof(uint32_t) == MLIRPSP::MIPSStateLayout::LlBit);

// Minimal platform boundary used by PPSSPP's headless unit-test executable.
// Keeping these in the external adapter avoids modifying the PPSSPP submodule.
void System_LaunchUrl(LaunchUrlType, std::string_view) {}
std::string System_GetProperty(SystemProperty) { return ""; }
std::vector<std::string> System_GetPropertyStringVec(SystemProperty) { return {}; }
int64_t System_GetPropertyInt(SystemProperty prop) {
	return prop == SYSPROP_SYSTEMVERSION ? 31 : -1;
}
float System_GetPropertyFloat(SystemProperty) { return -1.0f; }
bool System_GetPropertyBool(SystemProperty prop) {
	return prop == SYSPROP_IS_HEADLESS || prop == SYSPROP_CAN_JIT;
}
void System_Notify(SystemNotification) {}
void System_PostUIMessage(UIMessage, std::string_view) {}
void System_RunOnMainThread(std::function<void()>) {}
std::vector<std::string> System_GetCameraDeviceList() { return {}; }
void System_AskForPermission(SystemPermission) {}
PermissionStatus System_GetPermissionStatus(SystemPermission) { return PERMISSION_STATUS_GRANTED; }
void System_AudioGetDebugStats(char *buf, size_t) { if (buf) buf[0] = '\0'; }
void System_AudioClear() {}
void System_AudioPushSamples(const s32 *, int, float) {}
bool System_MakeRequest(SystemRequestType, int, const std::string &, const std::string &,
	int64_t, int64_t) { return false; }
bool NativeSaveSecret(std::string_view, std::string_view) { return false; }
std::string NativeLoadSecret(std::string_view) { return ""; }

namespace {

bool ParseU32(const char *text, uint32_t &value) {
	int base = 10;
	const char *begin = text;
	if (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
		base = 16;
		begin += 2;
	}
	const char *end = text + std::strlen(text);
	auto parsed = std::from_chars(begin, end, value, base);
	return parsed.ec == std::errc{} && parsed.ptr == end;
}

bool InitializeCore() {
	coreState = CORE_RUNNING_CPU;
	currentMIPS = &mipsr4k;
	g_symbolMap = new SymbolMap();
	Memory::g_MemorySize = Memory::RAM_NORMAL_SIZE;
	PSP_CoreParameter().cpuCore = CPUCore::INTERPRETER;
	PSP_CoreParameter().fastForward = true;
	if (!Memory::Init(Memory::MemMapSetupFlags::Default))
		return false;
	mipsr4k.Reset();
	CoreTiming::Init(currentMIPS);
	InitVFPU();
	InitIR();
	return true;
}

void ShutdownCore() {
	CoreTiming::Shutdown();
	mipsr4k.Shutdown();
	Memory::Shutdown();
	coreState = CORE_POWERDOWN;
	currentMIPS = nullptr;
	delete g_symbolMap;
	g_symbolMap = nullptr;
}

}  // namespace

int main(int argc, char **argv) {
	if (argc < 3) {
		std::fprintf(stderr, "usage: ppsspp_reference_adapter <address> <mips-word>...\n");
		return 1;
	}
	uint32_t address = 0;
	if (!ParseU32(argv[1], address)) {
		std::fprintf(stderr, "invalid guest address\n");
		return 2;
	}
	std::vector<uint32_t> words;
	std::vector<std::pair<uint32_t, uint32_t>> initialMemory;
	std::vector<uint32_t> observedMemory;
	std::vector<std::pair<uint32_t, uint32_t>> initialRegisters;
	bool executeMLIR = false;
	bool benchmark = false;
	uint32_t repeatBlocks = 1;
	for (int i = 2; i < argc; ++i) {
		if (std::strcmp(argv[i], "--benchmark") == 0) {
			benchmark = true;
			continue;
		}
		if (std::strcmp(argv[i], "--execute-mlir") == 0) {
#ifdef MLIR_PSP_ADAPTER_ENABLE_JIT
			executeMLIR = true;
			continue;
#else
			std::fprintf(stderr, "adapter was built without MLIR JIT support\n");
			return 3;
#endif
		}
		if (std::strcmp(argv[i], "--repeat-block") == 0 && i + 1 < argc) {
			if (!ParseU32(argv[++i], repeatBlocks) || repeatBlocks == 0)
				return 3;
			continue;
		}
		if (std::strcmp(argv[i], "--memory") == 0 && i + 2 < argc) {
			uint32_t memoryAddress = 0, memoryValue = 0;
			if (!ParseU32(argv[++i], memoryAddress) || !ParseU32(argv[++i], memoryValue))
				return 3;
			initialMemory.emplace_back(memoryAddress, memoryValue);
			continue;
		}
		if (std::strcmp(argv[i], "--observe") == 0 && i + 1 < argc) {
			uint32_t memoryAddress = 0;
			if (!ParseU32(argv[++i], memoryAddress)) return 3;
			observedMemory.push_back(memoryAddress);
			continue;
		}
		if (std::strcmp(argv[i], "--reg") == 0 && i + 2 < argc) {
			uint32_t reg = 0, regValue = 0;
			if (!ParseU32(argv[++i], reg) ||
					reg >= MLIRPSP::MIPSStateLayout::StateWords ||
					!ParseU32(argv[++i], regValue))
				return 3;
			initialRegisters.emplace_back(reg, regValue);
			continue;
		}
		uint32_t word = 0;
		if (!ParseU32(argv[i], word)) {
			std::fprintf(stderr, "invalid MIPS word: %s\n", argv[i]);
			return 3;
		}
		words.push_back(word);
	}
	if (benchmark && !executeMLIR) {
		std::fprintf(stderr, "--benchmark requires --execute-mlir\n");
		return 3;
	}
	if (!InitializeCore()) {
		std::fprintf(stderr, "PPSSPP memory initialization failed\n");
		return 4;
	}

	uint32_t *guestCode = reinterpret_cast<uint32_t *>(Memory::GetPointerWriteUnchecked(address));
	std::memcpy(guestCode, words.data(), words.size() * sizeof(uint32_t));
	auto memoryCallbacks = MLIRPSP::CreatePPSSPPMemoryAccessCallbacks();
	for (const auto &[memoryAddress, memoryValue] : initialMemory) {
		if (memoryCallbacks.write(memoryCallbacks.userdata, memoryAddress, 4,
				memoryValue) != MLIRPSP::MemoryAccessStatus::Success) {
			std::fprintf(stderr, "invalid initial memory address: 0x%08X\n", memoryAddress);
			ShutdownCore();
			return 5;
		}
	}

	MIPSComp::IRFrontend frontend(true);
	IROptions options{};
	options.unalignedLoadStore = true;
	options.unalignedLoadStoreVec4 = false;
	options.preferVec4 = true;
	frontend.SetOptions(options);
	std::vector<IRInst> instructions;
	uint32_t mipsBytes = 0;
	frontend.DoJit(address, instructions, mipsBytes);

#ifdef MLIR_PSP_ADAPTER_ENABLE_JIT
	std::array<uint32_t, MLIRPSP::MIPSStateLayout::StateWords> mlirStateWords{};
	uint32_t mlirNextPC = 0;
	int32_t mlirDowncount = 0;
	bool mlirUsedFallback = false;
	std::vector<std::pair<uint32_t, uint32_t>> mlirObservedMemory;
	if (executeMLIR) {
		std::vector<uint8_t> ramSnapshot(Memory::g_MemorySize);
		std::memcpy(ramSnapshot.data(), Memory::GetPointerUnchecked(0x08000000),
			ramSnapshot.size());

		MIPSState mlirState;
		mlirState.Init();
		mlirState.pc = address;
		mlirState.downcount = 1000;
		for (const auto &[reg, regValue] : initialRegisters)
			reinterpret_cast<uint32_t *>(&mlirState)[reg] = regValue;

		MLIRPSP::PPSSPPMLIRRuntime runtime(&mlirState);
		if (benchmark) {
			auto prepared = runtime.PrepareBlock(address);
			if (!prepared || prepared.backend != MLIRPSP::RuntimeBlockBackend::MLIR) {
				std::fprintf(stderr, "benchmark requires a compiled MLIR block\n");
				mlirState.Shutdown();
				ShutdownCore();
				return 8;
			}
		}
		const auto benchmarkStart = std::chrono::steady_clock::now();
		auto dispatch = runtime.RunBlocks(repeatBlocks);
		const auto benchmarkEnd = std::chrono::steady_clock::now();
		if (benchmark)
			std::fprintf(stderr, "MLIR_BENCH blocks=%u warm_ms=%.3f\n", repeatBlocks,
				std::chrono::duration<double, std::milli>(benchmarkEnd - benchmarkStart).count());
		if (!dispatch || dispatch.blocksExecuted != repeatBlocks ||
				(repeatBlocks > 1 && dispatch.cacheHits != repeatBlocks - (benchmark ? 0 : 1))) {
			std::fprintf(stderr, "MLIR dispatch failed after %llu blocks: %s\n",
				static_cast<unsigned long long>(dispatch.blocksExecuted),
				dispatch.error.c_str());
			mlirState.Shutdown();
			ShutdownCore();
			return 8;
		}
		mlirNextPC = mlirState.pc;
		mlirUsedFallback = dispatch.fallbackBlocks != 0;
		mlirDowncount = mlirState.downcount;
		std::memcpy(mlirStateWords.data(), &mlirState, sizeof(mlirStateWords));
		for (uint32_t memoryAddress : observedMemory) {
			auto observed = memoryCallbacks.read(memoryCallbacks.userdata,
				memoryAddress, 4, false);
			if (observed.status != MLIRPSP::MemoryAccessStatus::Success) {
				mlirState.Shutdown();
				ShutdownCore();
				return 9;
			}
			mlirObservedMemory.emplace_back(memoryAddress, observed.value);
		}
		mlirState.Shutdown();
		std::memcpy(Memory::GetPointerWriteUnchecked(0x08000000),
			ramSnapshot.data(), ramSnapshot.size());
	}
#endif

	MIPSState referenceState;
	referenceState.Init();
	referenceState.pc = address;
	referenceState.downcount = 1000;
	for (const auto &[reg, regValue] : initialRegisters)
		reinterpret_cast<uint32_t *>(&referenceState)[reg] = regValue;
	uint32_t nextPC = referenceState.pc;
	for (uint32_t block = 0; block < repeatBlocks; ++block) {
		nextPC = IRInterpret(&referenceState, instructions.data());
		referenceState.pc = nextPC;
	}

#ifdef MLIR_PSP_ADAPTER_ENABLE_JIT
	if (executeMLIR) {
		const auto *referenceStateWords = reinterpret_cast<const uint32_t *>(&referenceState);
		auto compareRange = [&](size_t begin, size_t end) {
			for (size_t index = begin; index < end; ++index) {
				if (mlirStateWords[index] != referenceStateWords[index]) {
					std::fprintf(stderr, "MLIR/PPSSPP register mismatch at r%zu\n", index);
					return false;
				}
			}
			return true;
		};
		// Compare architectural registers, not PPSSPP's interpreter/JIT scratch words.
		if (!compareRange(MLIRPSP::MIPSStateLayout::GPRBase, 192) ||
				!compareRange(208, 224) || !compareRange(242, 246) ||
				!compareRange(MLIRPSP::MIPSStateLayout::LlBit,
					MLIRPSP::MIPSStateLayout::LlBit + 1)) {
			referenceState.Shutdown();
			ShutdownCore();
			return 10;
		}
		if (mlirNextPC != nextPC || mlirDowncount != referenceState.downcount) {
			std::fprintf(stderr, "MLIR/PPSSPP exit-state mismatch\n");
			referenceState.Shutdown();
			ShutdownCore();
			return 11;
		}
		for (const auto &[memoryAddress, mlirValue] : mlirObservedMemory) {
			auto observed = memoryCallbacks.read(memoryCallbacks.userdata,
				memoryAddress, 4, false);
			if (observed.status != MLIRPSP::MemoryAccessStatus::Success ||
					observed.value != mlirValue) {
				std::fprintf(stderr, "MLIR/PPSSPP memory mismatch at 0x%08X\n",
					memoryAddress);
				referenceState.Shutdown();
				ShutdownCore();
				return 12;
			}
		}
		std::printf(mlirUsedFallback ? "MLIR_FALLBACK_OK\n" : "MLIR_OK\n");
	}
#endif

	std::printf("MIPS_BYTES %u\n", mipsBytes);
	for (const IRInst &inst : instructions) {
		std::printf("IR %u %u %u %u %u\n", static_cast<unsigned>(inst.op),
			static_cast<unsigned>(inst.dest), static_cast<unsigned>(inst.src1),
			static_cast<unsigned>(inst.src2), inst.constant);
	}
	std::printf("PC %u\n", nextPC);
	std::printf("DOWNCOUNT %d\n", referenceState.downcount);
	const auto *referenceStateWords = reinterpret_cast<const uint32_t *>(&referenceState);
	for (size_t i = 0; i < MLIRPSP::MIPSStateLayout::StateWords; ++i)
		std::printf("REG %zu %u\n", i, referenceStateWords[i]);
	for (uint32_t memoryAddress : observedMemory) {
		auto observed = memoryCallbacks.read(memoryCallbacks.userdata,
			memoryAddress, 4, false);
		if (observed.status != MLIRPSP::MemoryAccessStatus::Success) {
			std::fprintf(stderr, "invalid observed memory address: 0x%08X\n", memoryAddress);
			referenceState.Shutdown();
			ShutdownCore();
			return 6;
		}
		std::printf("MEM %u %u\n", memoryAddress, observed.value);
	}

	referenceState.Shutdown();
	ShutdownCore();
	return 0;
}
