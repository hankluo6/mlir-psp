#pragma once

#include <string>
#include <vector>

#include "Core/MIPS/IR/IRInst.h"
#include "mlir/IR/BuiltinOps.h"

namespace MLIRPSP {

enum class CompileStatus {
	Compiled,
	Unsupported,
	Error,
};

// First-stage PPSSPP IR to MLIR lowering. The state argument is temporarily a
// word-addressed memref so the frontend can be tested independently of the JIT
// ABI. The LLVM lowering stage will replace it with MIPSState *.
struct CompileResult {
	CompileStatus status = CompileStatus::Error;
	mlir::OwningOpRef<mlir::ModuleOp> module;
	std::string error;
	std::string functionName;
	size_t supportedInstructions = 0;

	explicit operator bool() const { return status == CompileStatus::Compiled && !!module; }
};

CompileResult CompileBlockToMLIR(mlir::MLIRContext &context,
	const std::vector<IRInst> &instructions, u32 guestAddress);

bool IsSupportedForInitialLowering(IROp op);

}  // namespace MLIRPSP
