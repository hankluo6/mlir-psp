#include "mlir_psp/PPSSPPMemoryAccess.h"

#include "Core/MemMap.h"

namespace MLIRPSP {
namespace {

bool IsValidAccess(uint32_t address, uint32_t size) {
	// IRFrontend may combine MIPS LWL/LWR or SWL/SWR pairs into an unaligned
	// Load32/Store32 when unalignedLoadStore is enabled.  Such accesses are
	// intentional, so validate the complete byte range rather than imposing
	// natural alignment here.
	return (size == 1 || size == 2 || size == 4) &&
		Memory::IsValidRange(address, size);
}

MemoryReadResult ReadPPSSPPMemory(void *, uint32_t address, uint32_t size,
		bool signExtend) {
	if (!IsValidAccess(address, size))
		return {0, MemoryAccessStatus::InvalidAddress};

	uint32_t value = 0;
	switch (size) {
	case 1:
		value = Memory::ReadUnchecked_U8(address);
		if (signExtend)
			value = static_cast<uint32_t>(static_cast<int32_t>(
				static_cast<int8_t>(value)));
		break;
	case 2:
		value = Memory::ReadUnchecked_U16(address);
		if (signExtend)
			value = static_cast<uint32_t>(static_cast<int32_t>(
				static_cast<int16_t>(value)));
		break;
	case 4:
		value = Memory::ReadUnchecked_U32(address);
		break;
	}
	return {value, MemoryAccessStatus::Success};
}

MemoryAccessStatus WritePPSSPPMemory(void *, uint32_t address, uint32_t size,
		uint32_t value) {
	if (!IsValidAccess(address, size))
		return MemoryAccessStatus::InvalidAddress;

	switch (size) {
	case 1:
		Memory::WriteUnchecked_U8(static_cast<uint8_t>(value), address);
		break;
	case 2:
		Memory::WriteUnchecked_U16(static_cast<uint16_t>(value), address);
		break;
	case 4:
		Memory::WriteUnchecked_U32(value, address);
		break;
	}
	return MemoryAccessStatus::Success;
}

}  // namespace

MemoryAccessCallbacks CreatePPSSPPMemoryAccessCallbacks() {
	return {nullptr, &ReadPPSSPPMemory, &WritePPSSPPMemory};
}

}  // namespace MLIRPSP
