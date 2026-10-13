#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "mlir_psp/MLIRJitEngine.h"

struct MIPSState;

namespace MLIRPSP {

enum class RuntimeBlockBackend {
	MLIR,
	IRInterpreter,
};

struct RuntimeBlockResult {
	RuntimeBlockBackend backend = RuntimeBlockBackend::IRInterpreter;
	ExecuteResult execution;
	uint32_t mipsBytes = 0;
	bool cacheHit = false;
	std::string error;

	explicit operator bool() const {
		return error.empty() && static_cast<bool>(execution) &&
			execution.memoryStatus == MemoryAccessStatus::Success;
	}
};

struct RuntimePrepareResult {
	RuntimeBlockBackend backend = RuntimeBlockBackend::IRInterpreter;
	uint32_t mipsBytes = 0;
	bool cacheHit = false;
	std::string error;

	explicit operator bool() const { return error.empty(); }
};

enum class RuntimeStopReason {
	BlockLimit,
	DowncountExpired,
	CoreStateChanged,
	InvalidPC,
	ExecutionError,
};

struct RuntimeDispatchResult {
	RuntimeStopReason stopReason = RuntimeStopReason::BlockLimit;
	uint64_t blocksExecuted = 0;
	uint64_t mlirBlocks = 0;
	uint64_t fallbackBlocks = 0;
	uint64_t cacheHits = 0;
	std::string error;

	explicit operator bool() const { return error.empty(); }
};

// PPSSPP-specific runtime boundary. It forms IR blocks on cache misses,
// executes supported blocks through MLIR, and retains whole-block IR fallback
// for operations not yet implemented by the MLIR lowering.
class PPSSPPMLIRRuntime {
public:
	explicit PPSSPPMLIRRuntime(MIPSState *state);
	~PPSSPPMLIRRuntime();

	PPSSPPMLIRRuntime(const PPSSPPMLIRRuntime &) = delete;
	PPSSPPMLIRRuntime &operator=(const PPSSPPMLIRRuntime &) = delete;

	RuntimePrepareResult PrepareBlock(uint32_t guestAddress);
	RuntimeBlockResult ExecuteOneBlock();
	RuntimeDispatchResult RunBlocks(uint32_t maxBlocks);
	RuntimeDispatchResult RunLoopUntil(uint64_t globalTicks);
	void ClearCache();
	void InvalidateCache(uint32_t guestAddress, uint32_t length);
	size_t CachedBlockCount() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace MLIRPSP
