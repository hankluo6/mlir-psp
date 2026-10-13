#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "MIPSTestCaseLoader.h"
#include "mlir_psp/MLIRBlockCompiler.h"
#include "mlir_psp/MLIRJitEngine.h"
#include "mlir_psp/MLIRJitDriver.h"
#include "mlir_psp/MIPSStateLayout.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

namespace {

FILE *OpenPipe(const char *command) {
#if defined(_WIN32)
	return _popen(command, "r");
#else
	return popen(command, "r");
#endif
}

int ClosePipe(FILE *pipe) {
#if defined(_WIN32)
	return _pclose(pipe);
#else
	return pclose(pipe);
#endif
}

bool RunCompilerRegressionTests() {
	mlir::MLIRContext context;
	std::vector<IRInst> instructions{
		{IROp::Downcount, {0}, 0, 0, 7},
		{IROp::SetConst, {MIPS_REG_V0}, 0, 0, 22},
		{IROp::ExitToConst, {0}, 0, 0, 0x08901410},
		// PPSSPP can leave unreachable operations after an exit.
		{IROp::SetConst, {MIPS_REG_V1}, 0, 0, 1},
	};
	auto compiled = MLIRPSP::CompileBlockToMLIR(context, instructions, 0x089013F4);
	if (!compiled) {
		std::cerr << "IR-after-exit regression: " << compiled.error << '\n';
		return false;
	}
	return true;
}

bool RunMIPSTestCase(const MIPSTestCase &testCase, const std::filesystem::path &adapter,
		bool dumpMLIR) {
	std::string command = "\"" + adapter.string() + "\" " +
		std::to_string(testCase.guestAddress);
	for (uint32_t word : testCase.mipsWords)
		command += " " + std::to_string(word);
	for (const auto &[address, value] : testCase.initialMemory)
		command += " --memory " + std::to_string(address) + " " + std::to_string(value);
	for (const auto &[address, value] : testCase.expectedMemory)
		command += " --observe " + std::to_string(address);
	for (const auto &[reg, value] : testCase.initialRegisters)
		command += " --reg " + std::to_string(static_cast<unsigned>(reg)) + " " +
			std::to_string(value);
	FILE *pipe = OpenPipe(command.c_str());
	if (!pipe) {
		std::cerr << testCase.name << ": could not start PPSSPP adapter\n";
		return false;
	}
	std::array<uint32_t, MLIRPSP::MIPSStateLayout::StateWords> registers{};
	uint32_t pc = 0;
	int32_t downcount = 0;
	std::vector<IRInst> instructions;
	std::unordered_map<uint32_t, uint32_t> referenceMemory;
	char line[256];
	while (std::fgets(line, sizeof(line), pipe)) {
		unsigned index = 0, value = 0;
		if (std::sscanf(line, "REG %u %u", &index, &value) == 2 && index < registers.size())
			registers[index] = value;
		else if (std::sscanf(line, "PC %u", &value) == 1)
			pc = value;
		else {
			unsigned memoryAddress = 0, memoryValue = 0;
			if (std::sscanf(line, "MEM %u %u", &memoryAddress, &memoryValue) == 2) {
				referenceMemory[memoryAddress] = memoryValue;
				continue;
			}
			if (std::sscanf(line, "DOWNCOUNT %d", &downcount) == 1) {
			// Parsed above; retained for comparison with the generated code.
				continue;
			}
			unsigned op = 0, dest = 0, src1 = 0, src2 = 0, constant = 0;
			if (std::sscanf(line, "IR %u %u %u %u %u", &op, &dest, &src1,
					&src2, &constant) == 5) {
				instructions.push_back({static_cast<IROp>(op),
					{static_cast<IRReg>(dest)}, static_cast<IRReg>(src1),
					static_cast<IRReg>(src2), constant});
			}
		}
	}
	if (ClosePipe(pipe) != 0) {
		std::cerr << testCase.name << ": PPSSPP adapter failed\n";
		return false;
	}
	if (instructions.empty()) {
		std::cerr << testCase.name << ": PPSSPP produced no IR\n";
		return false;
	}

	mlir::MLIRContext context;
	auto compiled = MLIRPSP::CompileBlockToMLIR(context, instructions,
		testCase.guestAddress);
	if (!compiled) {
		if (compiled.status == MLIRPSP::CompileStatus::Unsupported) {
			for (const auto &[reg, expected] : testCase.expectedRegisters) {
				if (registers[reg] != expected) {
					std::cerr << testCase.name << ": PPSSPP fallback result differs at r"
						<< static_cast<unsigned>(reg) << '\n';
					return false;
				}
			}
			if (pc != testCase.expectedPC) {
				std::cerr << testCase.name << ": PPSSPP fallback PC differs\n";
				return false;
			}
			std::cout << "FALLBACK " << testCase.name << ": " << compiled.error << '\n';
			return true;
		}
		std::cerr << testCase.name << ": generated PPSSPP IR: " << compiled.error << '\n';
		return false;
	}
	if (dumpMLIR) {
		llvm::outs() << "// " << testCase.name << " (from PPSSPP frontend)\n";
		compiled.module->print(llvm::outs());
		llvm::outs() << "\n";
	}
	std::array<uint32_t, MLIRPSP::MIPSStateLayout::StateWords> jitState{};
	jitState[MLIRPSP::MIPSStateLayout::Downcount] = 1000;
	for (const auto &[reg, value] : testCase.initialRegisters)
		jitState[reg] = value;
	std::vector<uint32_t> jitMemory(0x02000000 / sizeof(uint32_t));
	for (const auto &[address, value] : testCase.initialMemory)
		jitMemory[(address & 0x01FFFFFC) >> 2] = value;
	MLIRPSP::MLIRJitDriver driver;
	const uint32_t mipsBytes = static_cast<uint32_t>(
		testCase.mipsWords.size() * sizeof(uint32_t));
	auto driven = driver.ExecuteBlock(testCase.guestAddress, mipsBytes, instructions,
		jitState.data(), jitState.size(), jitMemory.data(), jitMemory.size());
	if (!driven) {
		std::cerr << testCase.name << ": " << driven.error << '\n';
		return false;
	}
	if (driven.execution.memoryStatus != MLIRPSP::MemoryAccessStatus::Success) {
		std::cerr << testCase.name << ": MLIR memory helper failed\n";
		return false;
	}
	for (size_t index = 0; index < 32; ++index) {
		if (jitState[index] != registers[index]) {
			std::cerr << testCase.name << ": MLIR and PPSSPP differ at r" << index
				<< " (" << jitState[index] << " vs " << registers[index] << ")\n";
			return false;
		}
	}
	if (driven.execution.nextPC != pc || static_cast<int32_t>(
			jitState[MLIRPSP::MIPSStateLayout::Downcount]) != downcount) {
		std::cerr << testCase.name << ": MLIR and PPSSPP exit state differs\n";
		return false;
	}
	for (const auto &[address, expected] : testCase.expectedMemory) {
		uint32_t actual = jitMemory[(address & 0x01FFFFFC) >> 2];
		if (referenceMemory[address] != expected || actual != referenceMemory[address]) {
			std::cerr << testCase.name << ": memory differs at " << address << '\n';
			return false;
		}
	}
	for (const auto &[reg, expected] : testCase.expectedRegisters) {
		if (registers[reg] != expected) {
			std::cerr << testCase.name << ": PPSSPP r" << static_cast<unsigned>(reg)
				<< " expected " << expected << ", got " << registers[reg] << '\n';
			return false;
		}
	}
	if (pc != testCase.expectedPC) {
		std::cerr << testCase.name << ": PPSSPP expected PC " << testCase.expectedPC
			<< ", got " << pc << '\n';
		return false;
	}

	// Execute the same address from identical inputs again. This must reuse the
	// persistent ExecutionEngine while binding fresh state and memory memrefs.
	std::array<uint32_t, MLIRPSP::MIPSStateLayout::StateWords> cachedState{};
	cachedState[MLIRPSP::MIPSStateLayout::Downcount] = 1000;
	for (const auto &[reg, value] : testCase.initialRegisters)
		cachedState[reg] = value;
	std::vector<uint32_t> cachedMemory(0x02000000 / sizeof(uint32_t));
	for (const auto &[address, value] : testCase.initialMemory)
		cachedMemory[(address & 0x01FFFFFC) >> 2] = value;
	auto cached = driver.ExecuteBlock(testCase.guestAddress, mipsBytes, instructions,
		cachedState.data(), cachedState.size(), cachedMemory.data(), cachedMemory.size());
	if (!cached || !cached.cacheHit || cached.execution.nextPC != driven.execution.nextPC ||
			cachedState != jitState || cachedMemory != jitMemory) {
		std::cerr << testCase.name << ": cached MLIR block execution differs\n";
		return false;
	}
	if (driver.CachedBlockCount() != 1) {
		std::cerr << testCase.name << ": unexpected MLIR block cache size\n";
		return false;
	}
	driver.InvalidateCache(testCase.guestAddress + 4, 4);
	if (driver.CachedBlockCount() != 0) {
		std::cerr << testCase.name << ": overlapping MLIR block was not invalidated\n";
		return false;
	}
	return true;
}

}  // namespace

int main(int argc, char **argv) {
	if (!RunCompilerRegressionTests())
		return 5;
    if (argc > 1) {
        std::string firstArg = argv[1];
        if (firstArg == "-h" || firstArg == "--help") {
            std::cout << "usage: mlir_psp_test <fixture-directory> [--dump] <adapter>\n";
            return 0;
        }
        if (firstArg[0] == '-' && firstArg != "--dump") {
            std::cerr << "invalid argument: " << firstArg << '\n';
            std::cerr << "usage: mlir_psp_test <fixture-directory> [--dump] <adapter>\n";
            return 1;
        }
    }
    if (argc < 2) {
        std::cerr << "usage: mlir_psp_test <fixture-directory> [--dump] <adapter>\n";
        return 1;
    }
	bool dumpMLIR = false;
	std::filesystem::path adapter;
	if (argc > 2) {
		std::string secondArg = argv[2];
		if (secondArg == "--dump" || secondArg == "--no-dump") {
			dumpMLIR = secondArg == "--dump";
			if (argc > 3)
				adapter = argv[3];
		} else {
			adapter = argv[2];
		}
	}
	if (adapter.empty()) {
		std::cerr << "a PPSSPP adapter is required for .mipstest fixtures\n";
		return 1;
	}
    std::vector<std::filesystem::path> fixtures;
    for (const auto &entry : std::filesystem::directory_iterator(argv[1])) {
        if (entry.path().extension() == ".mipstest")
            fixtures.push_back(entry.path());
    }
    std::sort(fixtures.begin(), fixtures.end());
    if (fixtures.empty()) {
        std::cerr << "no .mipstest fixtures found in " << argv[1] << '\n';
        return 2;
    }

    for (const auto &path : fixtures) {
        MIPSTestCase testCase;
        std::string error;
        if (!LoadMIPSTestCase(path, testCase, error)) {
            std::cerr << error << '\n';
            return 3;
        }
        if (!RunMIPSTestCase(testCase, adapter, dumpMLIR))
            return 4;
        std::cout << "PASS " << testCase.name << '\n';
    }
    return 0;
}
