#include "mlir_psp/PPSSPPMLIRJit.h"

#include <cstdio>
#include <cstdlib>

#include "Common/Log.h"
#include "Core/MIPS/JitCommon/JitCommon.h"

namespace MIPSComp {

// MIPS.cpp is redirected to this selector by the injected CMake integration.
// The bool is retained to match CreateNativeJit's ABI; an MLIR-enabled build
// deliberately selects MLIR for either PPSSPP native-JIT setting.
JitInterface *CreateMLIRJit(MIPSState *state, bool useIR) {
	if (const char *path = std::getenv("MLIR_PSP_TRACE_FILE")) {
		if (FILE *file = std::fopen(path, "a")) {
			std::fprintf(file, "selector entered state=%p useIR=%d\n", state,
				useIR ? 1 : 0);
			std::fclose(file);
		}
	}
	NOTICE_LOG(Log::JIT, "Creating MLIR JIT (requested PPSSPP IR mode=%d)",
		useIR ? 1 : 0);
	return MLIRPSP::CreatePPSSPPMLIRJit(state);
}

}  // namespace MIPSComp
