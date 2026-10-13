#include "MIPSTestCaseLoader.h"

#include <charconv>
#include <fstream>
#include <sstream>

namespace {

bool ParseU32(const std::string &text, uint32_t &value) {
	int base = 10;
	const char *begin = text.data();
	if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
		base = 16;
		begin += 2;
	}
	auto result = std::from_chars(begin, text.data() + text.size(), value, base);
	return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

bool ParseReg(const std::string &text, IRReg &reg) {
	static const std::unordered_map<std::string, IRReg> named{
		{"zero", MIPS_REG_ZERO}, {"v0", MIPS_REG_V0}, {"v1", MIPS_REG_V1},
		{"a0", MIPS_REG_A0}, {"a1", MIPS_REG_A1}, {"a2", MIPS_REG_A2},
		{"a3", MIPS_REG_A3}, {"t0", MIPS_REG_T0}, {"t1", MIPS_REG_T1},
		{"t2", MIPS_REG_T2}, {"t3", MIPS_REG_T3}, {"s0", MIPS_REG_S0},
		{"s1", MIPS_REG_S1}, {"sp", MIPS_REG_SP}, {"ra", MIPS_REG_RA},
	};
	auto found = named.find(text);
	if (found != named.end()) {
		reg = found->second;
		return true;
	}
	if (text.size() > 1 && text[0] == 'r') {
		uint32_t index = 0;
		if (ParseU32(text.substr(1), index) && index < 32) {
			reg = static_cast<IRReg>(index);
			return true;
		}
	}
	if (text.size() > 1 && text[0] == 'f') {
		uint32_t index = 0;
		if (ParseU32(text.substr(1), index) && index < 32) {
			reg = static_cast<IRReg>(32 + index);
			return true;
		}
	}
	return false;
}

}  // namespace

bool LoadMIPSTestCase(const std::filesystem::path &path, MIPSTestCase &testCase,
		std::string &error) {
	std::ifstream input(path);
	if (!input) {
		error = "cannot open fixture: " + path.string();
		return false;
	}
	testCase = {};
	testCase.name = path.stem().string();
	std::string line;
	for (size_t lineNumber = 1; std::getline(input, line); ++lineNumber) {
		auto comment = line.find('#');
		if (comment != std::string::npos)
			line.erase(comment);
		std::istringstream stream(line);
		std::string op;
		if (!(stream >> op))
			continue;

		auto fail = [&](const std::string &message) {
			error = path.string() + ":" + std::to_string(lineNumber) + ": " + message;
			return false;
		};
		if (op == "address" || op == "expect_pc" || op == "mips_word") {
			std::string number;
			uint32_t value = 0;
			if (!(stream >> number) || !ParseU32(number, value))
				return fail("expected a 32-bit integer");
			if (op == "address") testCase.guestAddress = value;
			if (op == "expect_pc") testCase.expectedPC = value;
			if (op == "mips_word") testCase.mipsWords.push_back(value);
			continue;
		}
		if (op == "memory" || op == "expect_memory") {
			std::string addressText, valueText;
			uint32_t address = 0, value = 0;
			if (!(stream >> addressText >> valueText) || !ParseU32(addressText, address) ||
					!ParseU32(valueText, value))
				return fail("expected a memory address and 32-bit value");
			(op == "memory" ? testCase.initialMemory : testCase.expectedMemory)[address] = value;
			continue;
		}
		if (op == "expect" || op == "initial") {
			std::string regText, valueText;
			IRReg reg = 0;
			uint32_t value = 0;
			if (!(stream >> regText >> valueText) || !ParseReg(regText, reg) ||
					!ParseU32(valueText, value))
				return fail("expected a register and 32-bit value");
			(op == "initial" ? testCase.initialRegisters : testCase.expectedRegisters)[reg] = value;
			continue;
		}
		return fail("unknown operation '" + op + "'");
	}
	if (testCase.mipsWords.empty()) {
		error = path.string() + ": fixture contains no MIPS instructions";
		return false;
	}
	return true;
}
