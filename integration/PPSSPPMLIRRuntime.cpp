#include "mlir_psp/PPSSPPMLIRRuntime.h"
#include "mlir_psp/Profiling.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "Common/Log.h"
#include "Core/MIPS/MIPS.h"
#include "Core/Core.h"
#include "Core/CoreTiming.h"
#include "Core/MemMap.h"
#include "Core/MIPS/IR/IRFrontend.h"
#include "Core/MIPS/IR/IRInterpreter.h"
#include "mlir_psp/MIPSStateLayout.h"
#include "mlir_psp/MLIRJitDriver.h"
#include "mlir_psp/PPSSPPMemoryAccess.h"

namespace MLIRPSP {
namespace {

void TraceMLIR(const char *format, ...) {
	const char *path = std::getenv("MLIR_PSP_TRACE_FILE");
	if (!path || !*path)
		return;
	FILE *file = std::fopen(path, "a");
	if (!file)
		return;
	va_list args;
	va_start(args, format);
	std::vfprintf(file, format, args);
	va_end(args);
	std::fputc('\n', file);
	std::fclose(file);
}

struct FallbackBlock {
	uint32_t guestAddress = 0;
	uint32_t mipsBytes = 0;
	std::vector<IRInst> instructions;
};

bool RangesOverlap(uint32_t firstAddress, uint32_t firstLength,
		uint32_t secondAddress, uint32_t secondLength) {
	const uint64_t firstEnd = static_cast<uint64_t>(firstAddress) + firstLength;
	const uint64_t secondEnd = static_cast<uint64_t>(secondAddress) + secondLength;
	return static_cast<uint64_t>(firstAddress) < secondEnd &&
		static_cast<uint64_t>(secondAddress) < firstEnd;
}

}  // namespace

struct PPSSPPMLIRRuntime::Impl {
	explicit Impl(MIPSState *mipsState)
			: state(mipsState), frontend(mipsState && mipsState->HasDefaultPrefix()),
			  memoryCallbacks(CreatePPSSPPMemoryAccessCallbacks()) {
		IROptions options{};
		options.unalignedLoadStore = true;
		options.unalignedLoadStoreVec4 = false;
		options.preferVec4 = true;
		frontend.SetOptions(options);
	}

	MIPSState *state = nullptr;
	MIPSComp::IRFrontend frontend;
	MemoryAccessCallbacks memoryCallbacks;
	MLIRJitDriver jit;
	std::unordered_map<uint32_t, FallbackBlock> fallbackBlocks;
	uint64_t compiledBlocks = 0;
	uint64_t fallbackBlocksPrepared = 0;
	uint64_t mlirExecutions = 0;
	uint64_t fallbackExecutions = 0;
	uint64_t cacheHits = 0;
	bool traceEnabled = [] {
		const char *path = std::getenv("MLIR_PSP_TRACE_FILE");
		return path && *path;
	}();
	std::unordered_map<uint32_t, uint64_t> executionsByPC;
	uint32_t traceSequenceRemaining = 0;
};

PPSSPPMLIRRuntime::PPSSPPMLIRRuntime(MIPSState *state)
	: impl_(std::make_unique<Impl>(state)) {
	TraceMLIR("runtime created state=%p", state);
	NOTICE_LOG(Log::JIT, "MLIR JIT runtime created (state=%p)", state);
}
PPSSPPMLIRRuntime::~PPSSPPMLIRRuntime() {
	TraceMLIR("summary compiled=%llu fallback=%llu mlir_exec=%llu fallback_exec=%llu cache_hits=%llu",
		static_cast<unsigned long long>(impl_->compiledBlocks),
		static_cast<unsigned long long>(impl_->fallbackBlocksPrepared),
		static_cast<unsigned long long>(impl_->mlirExecutions),
		static_cast<unsigned long long>(impl_->fallbackExecutions),
		static_cast<unsigned long long>(impl_->cacheHits));
	NOTICE_LOG(Log::JIT,
		"MLIR JIT summary: compiled=%llu fallback=%llu mlir_exec=%llu fallback_exec=%llu cache_hits=%llu",
		static_cast<unsigned long long>(impl_->compiledBlocks),
		static_cast<unsigned long long>(impl_->fallbackBlocksPrepared),
		static_cast<unsigned long long>(impl_->mlirExecutions),
		static_cast<unsigned long long>(impl_->fallbackExecutions),
		static_cast<unsigned long long>(impl_->cacheHits));
	if (impl_->traceEnabled) {
		std::vector<std::pair<uint32_t, uint64_t>> hottest(
			impl_->executionsByPC.begin(), impl_->executionsByPC.end());
		std::sort(hottest.begin(), hottest.end(), [](const auto &lhs, const auto &rhs) {
			return lhs.second > rhs.second;
		});
		for (size_t i = 0; i < std::min<size_t>(10, hottest.size()); ++i)
			TraceMLIR("hot_pc pc=%08x executions=%llu", hottest[i].first,
				static_cast<unsigned long long>(hottest[i].second));
	}
}

RuntimePrepareResult PPSSPPMLIRRuntime::PrepareBlock(uint32_t guestAddress) {
	RuntimePrepareResult result;
	if (!impl_->state) {
		result.error = "PPSSPP MLIR runtime requires MIPSState";
		return result;
	}

	if (impl_->jit.HasCachedBlock(guestAddress)) {
		result.backend = RuntimeBlockBackend::MLIR;
		result.cacheHit = true;
		++impl_->cacheHits;
		return result;
	}

	auto fallback = impl_->fallbackBlocks.find(guestAddress);
	if (fallback != impl_->fallbackBlocks.end()) {
		result.backend = RuntimeBlockBackend::IRInterpreter;
		result.cacheHit = true;
		result.mipsBytes = fallback->second.mipsBytes;
		++impl_->cacheHits;
		return result;
	}

	std::vector<IRInst> instructions;
	uint32_t mipsBytes = 0;
	{
		MLIR_PSP_PROFILE(Lift);
		impl_->frontend.DoJit(guestAddress, instructions, mipsBytes);
	}
	if (const char *tracePCText = std::getenv("MLIR_PSP_TRACE_PC")) {
		const bool traceAll = std::strcmp(tracePCText, "all") == 0;
		char *end = nullptr;
		const uint32_t tracePC = static_cast<uint32_t>(std::strtoul(tracePCText, &end, 16));
		if (traceAll || (end != tracePCText && guestAddress == tracePC)) {
			for (size_t index = 0; index < instructions.size(); ++index) {
				char text[256]{};
				DisassembleIR(text, sizeof(text), instructions[index]);
				TraceMLIR("trace_ir pc=%08x index=%llu %s", guestAddress,
					static_cast<unsigned long long>(index), text);
			}
		}
	}
	DEBUG_LOG(Log::JIT, "MLIR JIT lifted block %08x: %u MIPS bytes, %llu IR instructions",
		guestAddress, mipsBytes,
		static_cast<unsigned long long>(instructions.size()));
	result.mipsBytes = mipsBytes;
	auto prepared = impl_->jit.PrepareBlock(guestAddress, mipsBytes, instructions);
	if (prepared.compileStatus == CompileStatus::Unsupported) {
		TraceMLIR("fallback pc=%08x bytes=%u ir=%llu", guestAddress, mipsBytes,
			static_cast<unsigned long long>(instructions.size()));
		result.backend = RuntimeBlockBackend::IRInterpreter;
		FallbackBlock block{guestAddress, mipsBytes, std::move(instructions)};
		impl_->fallbackBlocks.emplace(guestAddress, std::move(block));
		++impl_->fallbackBlocksPrepared;
		INFO_LOG(Log::JIT, "MLIR JIT fallback block %08x (%u MIPS bytes)",
			guestAddress, mipsBytes);
		return result;
	}

	result.backend = RuntimeBlockBackend::MLIR;
	result.cacheHit = prepared.cacheHit;
	result.error = std::move(prepared.error);
	if (!result.error.empty()) {
		TraceMLIR("compile_error pc=%08x error=%s", guestAddress, result.error.c_str());
		for (size_t index = 0; index < instructions.size(); ++index) {
			char text[256]{};
			DisassembleIR(text, sizeof(text), instructions[index]);
			TraceMLIR("compile_error_ir pc=%08x index=%llu %s", guestAddress,
				static_cast<unsigned long long>(index), text);
		}
		ERROR_LOG(Log::JIT, "MLIR JIT compile failed at %08x: %s",
			guestAddress, result.error.c_str());
	} else {
		TraceMLIR("compiled pc=%08x bytes=%u ir=%llu", guestAddress, mipsBytes,
			static_cast<unsigned long long>(instructions.size()));
		++impl_->compiledBlocks;
		INFO_LOG(Log::JIT, "MLIR JIT compiled block %08x (%u MIPS bytes)",
			guestAddress, mipsBytes);
	}
	return result;
}

RuntimeBlockResult PPSSPPMLIRRuntime::ExecuteOneBlock() {
	RuntimeBlockResult result;
	if (!impl_->state) {
		result.error = "PPSSPP MLIR runtime requires MIPSState";
		result.execution.error = result.error;
		return result;
	}

	const uint32_t guestAddress = impl_->state->pc;
	if (impl_->traceEnabled)
		++impl_->executionsByPC[guestAddress];
	if (impl_->traceEnabled && impl_->traceSequenceRemaining > 0) {
		TraceMLIR("trace_sequence pc=%08x a1=%08x t0=%08x downcount=%d",
			guestAddress, impl_->state->r[MIPS_REG_A1], impl_->state->r[MIPS_REG_T0],
			impl_->state->downcount);
		--impl_->traceSequenceRemaining;
	}
	auto prepared = PrepareBlock(guestAddress);
	result.backend = prepared.backend;
	result.mipsBytes = prepared.mipsBytes;
	result.cacheHit = prepared.cacheHit;
	result.error = std::move(prepared.error);
	if (!result.error.empty()) {
		result.execution.error = result.error;
		return result;
	}

	if (prepared.backend == RuntimeBlockBackend::IRInterpreter) {
		auto fallback = impl_->fallbackBlocks.find(guestAddress);
		result.execution.nextPC = IRInterpret(impl_->state,
			fallback->second.instructions.data());
		++impl_->fallbackExecutions;
	} else {
		auto driven = impl_->jit.ExecuteCachedBlock(guestAddress,
			reinterpret_cast<uint32_t *>(impl_->state), MIPSStateLayout::StateWords,
			nullptr, 0, &impl_->memoryCallbacks);
		result.execution = std::move(driven.execution);
		result.error = std::move(driven.error);
		if (result.error.empty() &&
				result.execution.memoryStatus != MemoryAccessStatus::Success) {
			result.error = "MLIR block memory access failed with status " +
				std::to_string(static_cast<uint32_t>(result.execution.memoryStatus));
			result.execution.error = result.error;
		}
		++impl_->mlirExecutions;
		if (!result.error.empty()) {
			TraceMLIR("execute_error pc=%08x error=%s", guestAddress,
				result.error.c_str());
			ERROR_LOG(Log::JIT, "MLIR JIT execution failed at %08x: %s",
				guestAddress, result.error.c_str());
		}
	}
	if (result)
		impl_->state->pc = result.execution.nextPC;
	if (impl_->traceEnabled) {
		const char *tracePCText = std::getenv("MLIR_PSP_TRACE_PC");
		char *end = nullptr;
		const uint32_t tracePC = tracePCText
			? static_cast<uint32_t>(std::strtoul(tracePCText, &end, 16)) : 0;
		const uint64_t executions = impl_->executionsByPC[guestAddress];
		if (tracePCText && end != tracePCText && guestAddress == tracePC && executions <= 10) {
			TraceMLIR("trace_exec pc=%08x next=%08x state_pc=%08x a1=%08x t0=%08x t2=%08x t3=%08x downcount=%d",
				guestAddress, result.execution.nextPC, impl_->state->pc,
				impl_->state->r[MIPS_REG_A1],
				impl_->state->r[MIPS_REG_T0], impl_->state->r[MIPS_REG_T2],
				impl_->state->r[MIPS_REG_T3], impl_->state->downcount);
			if (executions == 1)
				impl_->traceSequenceRemaining = 10;
		}
	}
	return result;
}

RuntimeDispatchResult PPSSPPMLIRRuntime::RunBlocks(uint32_t maxBlocks) {
	RuntimeDispatchResult result;
	if (!impl_->state) {
		result.stopReason = RuntimeStopReason::ExecutionError;
		result.error = "PPSSPP MLIR runtime requires MIPSState";
		return result;
	}

	while (result.blocksExecuted < maxBlocks) {
		if (coreState != CORE_RUNNING_CPU) {
			result.stopReason = RuntimeStopReason::CoreStateChanged;
			return result;
		}
		if (impl_->state->downcount < 0) {
			result.stopReason = RuntimeStopReason::DowncountExpired;
			return result;
		}
		const uint32_t blockStart = impl_->state->pc;
		if (!Memory::IsValid4AlignedAddress(blockStart)) {
			Core_ExecException(blockStart, blockStart, ExecExceptionType::JUMP);
			result.stopReason = RuntimeStopReason::InvalidPC;
			return result;
		}

		auto block = ExecuteOneBlock();
		if (!block) {
			result.stopReason = RuntimeStopReason::ExecutionError;
			result.error = block.error.empty() ? block.execution.error : block.error;
			return result;
		}
		++result.blocksExecuted;
		if (block.backend == RuntimeBlockBackend::MLIR)
			++result.mlirBlocks;
		else
			++result.fallbackBlocks;
		if (block.cacheHit)
			++result.cacheHits;

		if (!Memory::IsValid4AlignedAddress(impl_->state->pc)) {
			Core_ExecException(impl_->state->pc, blockStart, ExecExceptionType::JUMP);
			result.stopReason = RuntimeStopReason::InvalidPC;
			return result;
		}
	}

	result.stopReason = RuntimeStopReason::BlockLimit;
	return result;
}

RuntimeDispatchResult PPSSPPMLIRRuntime::RunLoopUntil(uint64_t globalTicks) {
	// PPSSPP's current IRJit dispatcher also receives this value indirectly:
	// the caller configures CoreTiming's slice before entering RunLoopUntil().
	(void)globalTicks;
	RuntimeDispatchResult total;
	if (!impl_->state) {
		total.stopReason = RuntimeStopReason::ExecutionError;
		total.error = "PPSSPP MLIR runtime requires MIPSState";
		return total;
	}
	while (true) {
		CoreTiming::Advance(impl_->state);
		if (coreState != CORE_RUNNING_CPU) {
			total.stopReason = RuntimeStopReason::CoreStateChanged;
			return total;
		}

		auto slice = RunBlocks(std::numeric_limits<uint32_t>::max());
		total.blocksExecuted += slice.blocksExecuted;
		total.mlirBlocks += slice.mlirBlocks;
		total.fallbackBlocks += slice.fallbackBlocks;
		total.cacheHits += slice.cacheHits;
		if (!slice) {
			total.stopReason = slice.stopReason;
			total.error = std::move(slice.error);
			return total;
		}
		if (slice.stopReason != RuntimeStopReason::DowncountExpired) {
			total.stopReason = slice.stopReason;
			return total;
		}
	}
}

void PPSSPPMLIRRuntime::ClearCache() {
	impl_->jit.ClearCache();
	impl_->fallbackBlocks.clear();
}

void PPSSPPMLIRRuntime::InvalidateCache(uint32_t guestAddress, uint32_t length) {
	impl_->jit.InvalidateCache(guestAddress, length);
	if (length == 0)
		return;
	for (auto it = impl_->fallbackBlocks.begin(); it != impl_->fallbackBlocks.end();) {
		const auto &block = it->second;
		if (RangesOverlap(block.guestAddress, block.mipsBytes, guestAddress, length))
			it = impl_->fallbackBlocks.erase(it);
		else
			++it;
	}
}

size_t PPSSPPMLIRRuntime::CachedBlockCount() const {
	return impl_->jit.CachedBlockCount() + impl_->fallbackBlocks.size();
}

}  // namespace MLIRPSP
