#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir_psp/MemoryAccessABI.h"

namespace MLIRPSP {

struct ExecuteResult {
	uint32_t nextPC = 0;
	MemoryAccessStatus memoryStatus = MemoryAccessStatus::Success;
	std::string error;

	explicit operator bool() const { return error.empty(); }
};

struct CompileExecutableResult;

class CompiledMLIRBlock {
public:
	CompiledMLIRBlock();
	~CompiledMLIRBlock();
	CompiledMLIRBlock(CompiledMLIRBlock &&) noexcept;
	CompiledMLIRBlock &operator=(CompiledMLIRBlock &&) noexcept;

	CompiledMLIRBlock(const CompiledMLIRBlock &) = delete;
	CompiledMLIRBlock &operator=(const CompiledMLIRBlock &) = delete;

	ExecuteResult Execute(uint32_t *stateWords, size_t stateWordCount,
		uint32_t *memoryWords = nullptr, size_t memoryWordCount = 0,
		const MemoryAccessCallbacks *memoryCallbacks = nullptr);

private:
	struct Impl;
	explicit CompiledMLIRBlock(std::unique_ptr<Impl> impl);
	std::unique_ptr<Impl> impl_;

	friend CompileExecutableResult CompileToExecutable(mlir::ModuleOp module,
		const std::string &functionName);
};

struct CompileExecutableResult {
	std::unique_ptr<CompiledMLIRBlock> block;
	std::string error;

	explicit operator bool() const { return block != nullptr && error.empty(); }
};

CompileExecutableResult CompileToExecutable(mlir::ModuleOp module,
	const std::string &functionName);

}  // namespace MLIRPSP
