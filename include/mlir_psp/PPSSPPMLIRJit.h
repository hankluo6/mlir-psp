#pragma once

class MIPSState;

namespace MIPSComp {
class JitInterface;
}

namespace MLIRPSP {

// Creates the PPSSPP JitInterface bridge backed by PPSSPPMLIRRuntime.
// Ownership follows MIPSComp::CreateNativeJit(): the caller owns the result.
MIPSComp::JitInterface *CreatePPSSPPMLIRJit(MIPSState *state);

}  // namespace MLIRPSP
