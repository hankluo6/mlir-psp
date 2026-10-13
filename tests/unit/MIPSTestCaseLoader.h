#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "Core/MIPS/IR/IRInst.h"

struct MIPSTestCase {
	std::string name;
	uint32_t guestAddress = 0;
	std::vector<uint32_t> mipsWords;
	std::unordered_map<IRReg, uint32_t> initialRegisters;
	std::unordered_map<IRReg, uint32_t> expectedRegisters;
	std::unordered_map<uint32_t, uint32_t> initialMemory;
	std::unordered_map<uint32_t, uint32_t> expectedMemory;
	uint32_t expectedPC = 0;
};

bool LoadMIPSTestCase(const std::filesystem::path &path, MIPSTestCase &testCase,
	std::string &error);
