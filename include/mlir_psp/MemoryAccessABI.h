#pragma once

#include <cstdint>

namespace MLIRPSP {

// Contract used by the future slow path for addresses that cannot be handled
// by direct normal-RAM access.  The PPSSPP integration layer owns userdata and
// implements these callbacks using its Memory subsystem.
enum class MemoryAccessStatus : uint32_t {
	Success = 0,
	InvalidAddress = 1,
	Exception = 2,
	Breakpoint = 3,
	Fallback = 4,
};

struct MemoryReadResult {
	uint32_t value = 0;
	MemoryAccessStatus status = MemoryAccessStatus::Success;
};

using ReadMemoryCallback = MemoryReadResult (*)(void *userdata,
	uint32_t address, uint32_t size, bool signExtend);
using WriteMemoryCallback = MemoryAccessStatus (*)(void *userdata,
	uint32_t address, uint32_t size, uint32_t value);

struct MemoryAccessCallbacks {
	void *userdata = nullptr;
	ReadMemoryCallback read = nullptr;
	WriteMemoryCallback write = nullptr;
};

}  // namespace MLIRPSP
