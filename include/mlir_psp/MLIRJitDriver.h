#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "Core/MIPS/IR/IRInst.h"
#include "mlir_psp/MLIRBlockCompiler.h"
#include "mlir_psp/MLIRJitEngine.h"

namespace MLIRPSP {

struct DriverPrepareResult {
	CompileStatus compileStatus = CompileStatus::Error;
	bool cacheHit = false;
	std::string error;

	explicit operator bool() const {
		return compileStatus == CompileStatus::Compiled && error.empty();
	}
};

struct DriverExecutionResult {
	ExecuteResult execution;
	CompileStatus compileStatus = CompileStatus::Error;
	bool cacheHit = false;
	std::string error;

	explicit operator bool() const {
		return error.empty() && static_cast<bool>(execution);
	}
};

class MLIRJitDriver {
public:
	MLIRJitDriver();
	~MLIRJitDriver();
	MLIRJitDriver(MLIRJitDriver &&) noexcept;
	MLIRJitDriver &operator=(MLIRJitDriver &&) noexcept;

	MLIRJitDriver(const MLIRJitDriver &) = delete;
	MLIRJitDriver &operator=(const MLIRJitDriver &) = delete;

	DriverPrepareResult PrepareBlock(uint32_t guestAddress, uint32_t mipsBytes,
		const std::vector<IRInst> &instructions);
	DriverExecutionResult ExecuteBlock(uint32_t guestAddress, uint32_t mipsBytes,
		const std::vector<IRInst> &instructions, uint32_t *stateWords,
		size_t stateWordCount, uint32_t *memoryWords = nullptr,
		size_t memoryWordCount = 0,
		const MemoryAccessCallbacks *memoryCallbacks = nullptr);
	DriverExecutionResult ExecuteCachedBlock(uint32_t guestAddress,
		uint32_t *stateWords, size_t stateWordCount,
		uint32_t *memoryWords = nullptr, size_t memoryWordCount = 0,
		const MemoryAccessCallbacks *memoryCallbacks = nullptr);

	void ClearCache();
	void InvalidateCache(uint32_t guestAddress, uint32_t length);
	bool HasCachedBlock(uint32_t guestAddress) const;
	size_t CachedBlockCount() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};

}  // namespace MLIRPSP
