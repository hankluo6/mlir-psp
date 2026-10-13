#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <system_error>

using namespace llvm;

// Ignore NearBlock deliberately: this reproduces an allocator that places
// independent section groups more than 4 GiB apart.
class FarMapper final : public SectionMemoryManager::MemoryMapper {
  unsigned AllocationCount = 0;
public:
  sys::MemoryBlock allocateMappedMemory(SectionMemoryManager::AllocationPurpose,
                                        size_t Size, const sys::MemoryBlock *,
                                        unsigned, std::error_code &EC) override {
    const uintptr_t Hints[] = {0x1000000000ULL, 0x3000000000ULL,
                                0x5000000000ULL, 0x7000000000ULL};
    if (AllocationCount >= 4) {
      EC = std::make_error_code(std::errc::not_enough_memory);
      return {};
    }
    void *P = VirtualAlloc(reinterpret_cast<void *>(Hints[AllocationCount++]),
                           Size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!P) {
      EC = std::error_code(GetLastError(), std::system_category());
      return {};
    }
    return {P, Size};
  }
  std::error_code protectMappedMemory(const sys::MemoryBlock &B,
                                      unsigned Flags) override {
    DWORD Old;
    DWORD Mode = Flags & sys::Memory::MF_EXEC ? PAGE_EXECUTE_READ :
                 Flags & sys::Memory::MF_WRITE ? PAGE_READWRITE : PAGE_READONLY;
    if (!VirtualProtect(B.base(), B.allocatedSize(), Mode, &Old))
      return {static_cast<int>(GetLastError()), std::system_category()};
    return {};
  }
  std::error_code releaseMappedMemory(sys::MemoryBlock &B) override {
    if (!VirtualFree(B.base(), 0, MEM_RELEASE))
      return {static_cast<int>(GetLastError()), std::system_category()};
    return {};
  }
};

int main(int argc, char **argv) {
  if (argc != 3) return 2;
  bool Reserve = std::atoi(argv[1]) != 0;
  auto Buffer = MemoryBuffer::getFile(argv[2]);
  if (!Buffer) return 3;
  auto Obj = object::ObjectFile::createObjectFile(Buffer.get()->getMemBufferRef());
  if (!Obj) return 4;
  FarMapper Mapper;
  SectionMemoryManager MemMgr(&Mapper, Reserve);
  RuntimeDyld Dyld(MemMgr, MemMgr);
  Dyld.loadObject(**Obj);
  if (Dyld.hasError()) return 5;
  Dyld.resolveRelocations();
  if (Dyld.hasError()) return 6;
  std::string Err;
  if (MemMgr.finalizeMemory(&Err)) return 7;
  std::puts(Reserve ? "reserved layout passed" : "separate layout passed");
  return 0;
}
