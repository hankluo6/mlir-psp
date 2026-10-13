#include "mlir_psp/PPSSPPMLIRJit.h"

#include "Common/Log.h"
#include "Common/Serialize/Serializer.h"
#include "Core/Core.h"
#include "Core/MIPS/IR/IRJit.h"
#include "mlir_psp/PPSSPPMLIRRuntime.h"

namespace MLIRPSP {
namespace {

// IRJit supplies PPSSPP's existing debugger and emuhack bookkeeping contract.
// Execution and compilation are replaced with the external MLIR runtime.  The
// inherited IR block cache intentionally stays empty until a dedicated MLIR
// debugger cache is implemented.
class PPSSPPMLIRJit final : public MIPSComp::IRJit {
public:
	explicit PPSSPPMLIRJit(MIPSState *state)
		: MIPSComp::IRJit(state, true), runtime_(state) {
		NOTICE_LOG(Log::JIT, "MLIR JIT selected as PPSSPP CPU backend");
	}

	void RunLoopUntil(u64 globalTicks) override {
		auto result = runtime_.RunLoopUntil(globalTicks);
		if (!result) {
			ERROR_LOG(Log::JIT, "MLIR JIT run loop failed: %s", result.error.c_str());
			Core_UpdateState(CORE_RUNTIME_ERROR);
		}
	}

	void Compile(u32 guestAddress) override {
		auto result = runtime_.PrepareBlock(guestAddress);
		if (!result) {
			ERROR_LOG(Log::JIT, "MLIR JIT failed to compile block at %08x: %s",
				guestAddress, result.error.c_str());
			Core_UpdateState(CORE_RUNTIME_ERROR);
		}
	}

	void ClearCache() override {
		runtime_.ClearCache();
		MIPSComp::IRJit::ClearCache();
	}

	void InvalidateCacheAt(u32 guestAddress, int length = 4) override {
		if (length > 0)
			runtime_.InvalidateCache(guestAddress, static_cast<uint32_t>(length));
		MIPSComp::IRJit::InvalidateCacheAt(guestAddress, length);
	}

	void DoState(PointerWrap &p) override {
		MIPSComp::IRJit::DoState(p);
		// Loading can replace guest code, so cached native code is no longer
		// valid. Measuring/writing/verifying a save does not replace guest code:
		// retain the cache to avoid destroying engines and recompiling on save.
		if (p.mode == PointerWrap::MODE_READ)
			runtime_.ClearCache();
	}

private:
	PPSSPPMLIRRuntime runtime_;
};

}  // namespace

MIPSComp::JitInterface *CreatePPSSPPMLIRJit(MIPSState *state) {
	return new PPSSPPMLIRJit(state);
}

}  // namespace MLIRPSP
