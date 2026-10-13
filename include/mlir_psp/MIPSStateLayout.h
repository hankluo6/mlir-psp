#pragma once

#include <cstddef>

namespace MLIRPSP::MIPSStateLayout {

// Word offsets in the pinned PPSSPP MIPSState layout.  The external PPSSPP
// adapter validates these constants with offsetof() at compile time.
inline constexpr std::size_t StateWords = 256;
inline constexpr std::size_t GPRBase = 0;
inline constexpr std::size_t FPRBase = 32;
inline constexpr std::size_t VFPUBase = 64;
inline constexpr std::size_t VFPUCtrlBase = 208;
inline constexpr std::size_t PC = 241;
inline constexpr std::size_t Lo = 242;
inline constexpr std::size_t Hi = 243;
inline constexpr std::size_t FCR31 = 244;
inline constexpr std::size_t FPCond = 245;
inline constexpr std::size_t NextPC = 247;
inline constexpr std::size_t Downcount = 248;
inline constexpr std::size_t LlBit = 250;

}  // namespace MLIRPSP::MIPSStateLayout
