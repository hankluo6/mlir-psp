#pragma once

#include "mlir_psp/MemoryAccessABI.h"

namespace MLIRPSP {

// Creates callbacks backed by PPSSPP's currently initialized Memory subsystem.
// The returned callbacks do not own the PPSSPP runtime or its memory mapping.
MemoryAccessCallbacks CreatePPSSPPMemoryAccessCallbacks();

}  // namespace MLIRPSP
