#include "mlir_psp/MLIRBlockCompiler.h"
#include "mlir_psp/MIPSStateLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace MLIRPSP {
namespace {

constexpr int kStateWords = static_cast<int>(MIPSStateLayout::StateWords);
constexpr IRReg kDowncountWord = static_cast<IRReg>(MIPSStateLayout::Downcount);
constexpr IRReg kLoWord = static_cast<IRReg>(MIPSStateLayout::Lo);
constexpr IRReg kHiWord = static_cast<IRReg>(MIPSStateLayout::Hi);
constexpr IRReg kLlBitWord = static_cast<IRReg>(MIPSStateLayout::LlBit);

mlir::Value LoadReg(mlir::OpBuilder &builder, mlir::Location loc,
	mlir::Value state, IRReg reg) {
	if (reg == MIPS_REG_ZERO)
		return builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
	auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, reg);
	return builder.create<mlir::memref::LoadOp>(loc, state,
		mlir::ValueRange{index}, false, mlir::IntegerAttr{});
}

void StoreReg(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value state,
	IRReg reg, mlir::Value value) {
	if (reg == MIPS_REG_ZERO)
		return;
	auto index = builder.create<mlir::arith::ConstantIndexOp>(loc, reg);
	builder.create<mlir::memref::StoreOp>(loc, value, state,
		mlir::ValueRange{index}, false, mlir::IntegerAttr{});
}

}  // namespace

bool IsSupportedForInitialLowering(IROp op) {
	switch (op) {
	case IROp::SetConst:
	case IROp::SetConstF:
	case IROp::Mov:
	case IROp::Add:
	case IROp::Sub:
	case IROp::And:
	case IROp::Or:
	case IROp::Xor:
	case IROp::AddConst:
	case IROp::OptAddConst:
	case IROp::SubConst:
	case IROp::AndConst:
	case IROp::OrConst:
	case IROp::XorConst:
	case IROp::OptAndConst:
	case IROp::OptOrConst:
	case IROp::Shl:
	case IROp::Shr:
	case IROp::Sar:
	case IROp::ShlImm:
	case IROp::ShrImm:
	case IROp::SarImm:
	case IROp::Slt:
	case IROp::SltU:
	case IROp::SltConst:
	case IROp::SltUConst:
	case IROp::Load32:
	case IROp::Store32:
	case IROp::Load8:
	case IROp::Load8Ext:
	case IROp::Load16:
	case IROp::Load16Ext:
	case IROp::Store8:
	case IROp::Store16:
	case IROp::ValidateAddress8:
	case IROp::ValidateAddress16:
	case IROp::ValidateAddress32:
	case IROp::MtLo:
	case IROp::MtHi:
	case IROp::MfLo:
	case IROp::MfHi:
	case IROp::Mult:
	case IROp::MultU:
	case IROp::Madd:
	case IROp::MaddU:
	case IROp::Msub:
	case IROp::MsubU:
	case IROp::Div:
	case IROp::DivU:
	case IROp::Neg:
	case IROp::Not:
	case IROp::Ext8to32:
	case IROp::Ext16to32:
	case IROp::Ror:
	case IROp::RorImm:
	case IROp::Clz:
	case IROp::MovZ:
	case IROp::MovNZ:
	case IROp::Max:
	case IROp::Min:
	case IROp::BSwap16:
	case IROp::BSwap32:
	case IROp::ReverseBits:
	case IROp::Load32Left:
	case IROp::Load32Right:
	case IROp::Store32Left:
	case IROp::Store32Right:
	case IROp::Load32Linked:
	case IROp::LoadFloat:
	case IROp::LoadVec4:
	case IROp::Store32Conditional:
	case IROp::StoreFloat:
	case IROp::StoreVec4:
	case IROp::FMov:
	case IROp::FAbs:
	case IROp::FNeg:
	case IROp::FSign:
	case IROp::FAdd:
	case IROp::FSub:
	case IROp::FMul:
	case IROp::FDiv:
	case IROp::FMin:
	case IROp::FMax:
	case IROp::FSqrt:
	case IROp::FRound:
	case IROp::FTrunc:
	case IROp::FCeil:
	case IROp::FFloor:
	case IROp::FCvtWS:
	case IROp::FCvtSW:
	case IROp::FCvtScaledWS:
	case IROp::FCvtScaledSW:
	case IROp::OptFCvtSWFromGPR:
	case IROp::FSat0_1:
	case IROp::FSatMinus1_1:
	case IROp::FCmp:
	case IROp::FCmovVfpuCC:
	case IROp::FCmpVfpuBit:
	case IROp::FCmpVfpuAggregate:
	case IROp::RestoreRoundingMode:
	case IROp::ApplyRoundingMode:
	case IROp::UpdateRoundingMode:
	case IROp::FMovFromGPR:
	case IROp::FMovToGPR:
	case IROp::OptFMovToGPRShr8:
	case IROp::FpCondFromReg:
	case IROp::FpCondToReg:
	case IROp::FpCtrlFromReg:
	case IROp::FpCtrlToReg:
	case IROp::VfpuCtrlToReg:
	case IROp::SetCtrlVFPU:
	case IROp::SetCtrlVFPUReg:
	case IROp::SetCtrlVFPUFReg:
	case IROp::Vec4Init:
	case IROp::Vec4Shuffle:
	case IROp::Vec4Blend:
	case IROp::Vec4Mov:
	case IROp::Vec4Add:
	case IROp::Vec4Sub:
	case IROp::Vec4Mul:
	case IROp::Vec4Div:
	case IROp::Vec4Scale:
	case IROp::Vec4Dot:
	case IROp::Vec4Neg:
	case IROp::Vec4Abs:
	case IROp::Vec2Unpack16To31:
	case IROp::Vec2Unpack16To32:
	case IROp::Vec4Unpack8To32:
	case IROp::Vec4DuplicateUpperBitsAndShift1:
	case IROp::Vec4ClampToZero:
	case IROp::Vec2ClampToZero:
	case IROp::Vec4Pack31To8:
	case IROp::Vec4Pack32To8:
	case IROp::Vec2Pack31To16:
	case IROp::Vec2Pack32To16:
	case IROp::FSin:
	case IROp::FCos:
	case IROp::FRSqrt:
	case IROp::FRecip:
	case IROp::FAsin:
	case IROp::ExitToConst:
	case IROp::ExitToReg:
	case IROp::ExitToConstIfEq:
	case IROp::ExitToConstIfNeq:
	case IROp::ExitToConstIfGtZ:
	case IROp::ExitToConstIfGeZ:
	case IROp::ExitToConstIfLtZ:
	case IROp::ExitToConstIfLeZ:
	case IROp::ExitToConstIfFpTrue:
	case IROp::ExitToConstIfFpFalse:
	case IROp::SetPC:
	case IROp::SetPCConst:
	case IROp::ExitToPC:
	case IROp::Downcount:
	case IROp::Nop:
		return true;
	default:
		return false;
	}
}

CompileResult CompileBlockToMLIR(mlir::MLIRContext &context,
	const std::vector<IRInst> &instructions, u32 guestAddress) {
	CompileResult result;
	context.loadDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
		mlir::func::FuncDialect,
		mlir::memref::MemRefDialect>();

	for (const IRInst &inst : instructions) {
		if (!IsSupportedForInitialLowering(inst.op)) {
			result.status = CompileStatus::Unsupported;
			result.error = "unsupported PPSSPP IR operation #" +
				std::to_string(static_cast<unsigned>(inst.op));
			return result;
		}
		++result.supportedInstructions;
	}

	mlir::OpBuilder builder(&context);
	auto loc = builder.getUnknownLoc();
	auto module = mlir::ModuleOp::create(loc);
	auto i32 = builder.getI32Type();
	auto readMemory = mlir::func::FuncOp::create(loc, "mlir_psp_memory_read",
		builder.getFunctionType({i32, i32, i32}, {i32}));
	readMemory.setPrivate();
	module.push_back(readMemory);
	auto writeMemory = mlir::func::FuncOp::create(loc, "mlir_psp_memory_write",
		builder.getFunctionType({i32, i32, i32}, {}));
	writeMemory.setPrivate();
	module.push_back(writeMemory);
	auto writeMemoryConditional = mlir::func::FuncOp::create(loc,
		"mlir_psp_memory_write_conditional",
		builder.getFunctionType({i32, i32, i32, i32}, {i32}));
	writeMemoryConditional.setPrivate();
	module.push_back(writeMemoryConditional);
	auto floatSqrt = mlir::func::FuncOp::create(loc, "mlir_psp_float_sqrt",
		builder.getFunctionType({i32}, {i32}));
	floatSqrt.setPrivate();
	module.push_back(floatSqrt);
	auto floatBinary = mlir::func::FuncOp::create(loc, "mlir_psp_float_binary",
		builder.getFunctionType({i32, i32, i32}, {i32}));
	floatBinary.setPrivate();
	module.push_back(floatBinary);
	auto floatToInt = mlir::func::FuncOp::create(loc, "mlir_psp_float_to_int",
		builder.getFunctionType({i32, i32}, {i32}));
	floatToInt.setPrivate();
	module.push_back(floatToInt);
	auto floatUnary = mlir::func::FuncOp::create(loc, "mlir_psp_float_unary",
		builder.getFunctionType({i32, i32}, {i32}));
	floatUnary.setPrivate();
	module.push_back(floatUnary);
	auto floatToScaledInt = mlir::func::FuncOp::create(loc,
		"mlir_psp_float_to_scaled_int",
		builder.getFunctionType({i32, i32, i32}, {i32}));
	floatToScaledInt.setPrivate();
	module.push_back(floatToScaledInt);
	auto applyRounding = mlir::func::FuncOp::create(loc,
		"mlir_psp_apply_rounding", builder.getFunctionType({i32}, {}));
	applyRounding.setPrivate();
	module.push_back(applyRounding);
	auto restoreRounding = mlir::func::FuncOp::create(loc,
		"mlir_psp_restore_rounding", builder.getFunctionType({}, {}));
	restoreRounding.setPrivate();
	module.push_back(restoreRounding);
	auto stateType = mlir::MemRefType::get({kStateWords}, i32);
	auto memoryType = mlir::MemRefType::get({mlir::ShapedType::kDynamic}, i32);
	auto functionType = builder.getFunctionType({stateType, memoryType}, {i32});
	result.functionName = "ppsspp_block_" + std::to_string(guestAddress);
	auto function = mlir::func::FuncOp::create(loc, result.functionName, functionType);
	function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
	mlir::Block *entry = function.addEntryBlock();
	module.push_back(function);
	builder.setInsertionPointToStart(entry);
	mlir::Value state = entry->getArgument(0);
	mlir::Value memory = entry->getArgument(1);
	bool emittedReturn = false;
	bool blockTerminated = false;

	for (const IRInst &inst : instructions) {
		// PPSSPP may retain unreachable IR after a block exit.  A func.return is
		// an MLIR terminator, so no operation may be inserted after an
		// unconditional exit.  Conditional exits create a continuation block.
		if (blockTerminated)
			break;
		mlir::Value value;
		switch (inst.op) {
		case IROp::SetConst:
			value = builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32);
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		case IROp::SetConstF:
			value = builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest, value);
			break;
		case IROp::Mov:
			StoreReg(builder, loc, state, inst.dest, LoadReg(builder, loc, state, inst.src1));
			break;
		case IROp::FMov:
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1));
			break;
		case IROp::FSat0_1:
		case IROp::FSatMinus1_1: {
			auto inputBits = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			auto input = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), inputBits);
			auto minimumBits = builder.create<mlir::arith::ConstantIntOp>(loc,
				inst.op == IROp::FSat0_1 ? 0u : 0xBF800000u, 32);
			auto maximumBits = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x3F800000u, 32);
			auto minimum = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), minimumBits);
			auto maximum = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), maximumBits);
			auto atOrBelowMinimum = builder.create<mlir::arith::CmpFOp>(loc,
				mlir::arith::CmpFPredicate::OLE, input, minimum);
			auto atOrAboveMaximum = builder.create<mlir::arith::CmpFOp>(loc,
				mlir::arith::CmpFPredicate::OGE, input, maximum);
			auto belowClamped = builder.create<mlir::arith::SelectOp>(loc,
				atOrBelowMinimum, minimumBits, inputBits);
			auto clamped = builder.create<mlir::arith::SelectOp>(loc,
				atOrAboveMaximum, maximumBits, belowClamped);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				clamped);
			break;
		}
		case IROp::FAbs:
		case IROp::FNeg:
		case IROp::FSign: {
			auto input = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			auto signMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x80000000u, 32);
			if (inst.op == IROp::FAbs) {
				auto magnitudeMask = builder.create<mlir::arith::ConstantIntOp>(loc,
					0x7FFFFFFFu, 32);
				value = builder.create<mlir::arith::AndIOp>(loc, input, magnitudeMask);
			} else if (inst.op == IROp::FNeg) {
				value = builder.create<mlir::arith::XOrIOp>(loc, input, signMask);
			} else {
				auto magnitudeMask = builder.create<mlir::arith::ConstantIntOp>(loc,
					0x7FFFFFFFu, 32);
				auto magnitude = builder.create<mlir::arith::AndIOp>(loc, input,
					magnitudeMask);
				auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
				auto isZero = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::eq, magnitude, zero);
				auto sign = builder.create<mlir::arith::AndIOp>(loc, input, signMask);
				auto positiveOne = builder.create<mlir::arith::ConstantIntOp>(loc,
					0x3F800000u, 32);
				auto signedOne = builder.create<mlir::arith::OrIOp>(loc,
					positiveOne, sign);
				value = builder.create<mlir::arith::SelectOp>(loc, isZero, zero,
					signedOne);
			}
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				value);
			break;
		}
		case IROp::FAdd:
		case IROp::FSub:
		case IROp::FMul:
		case IROp::FDiv: {
			auto lhsBits = LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1);
			auto rhsBits = LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src2);
			uint32_t operation = 0;
			if (inst.op == IROp::FSub) operation = 1;
			if (inst.op == IROp::FMul) operation = 2;
			if (inst.op == IROp::FDiv) operation = 3;
			mlir::Value resultBits = builder.create<mlir::func::CallOp>(loc,
				floatBinary, mlir::ValueRange{lhsBits, rhsBits,
					builder.create<mlir::arith::ConstantIntOp>(loc, operation, 32)})
				.getResult(0);
			if (inst.op == IROp::FMul) {
				auto magnitudeMask = builder.create<mlir::arith::ConstantIntOp>(loc, 0x7FFFFFFFu, 32);
				auto lhsMagnitude = builder.create<mlir::arith::AndIOp>(loc, lhsBits, magnitudeMask);
				auto rhsMagnitude = builder.create<mlir::arith::AndIOp>(loc, rhsBits, magnitudeMask);
				auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
				auto infinity = builder.create<mlir::arith::ConstantIntOp>(loc, 0x7F800000u, 32);
				auto lhsZero = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::eq, lhsMagnitude, zero);
				auto rhsZero = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::eq, rhsMagnitude, zero);
				auto lhsInf = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::eq, lhsMagnitude, infinity);
				auto rhsInf = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::eq, rhsMagnitude, infinity);
				auto special = builder.create<mlir::arith::OrIOp>(loc,
					builder.create<mlir::arith::AndIOp>(loc, lhsZero, rhsInf),
					builder.create<mlir::arith::AndIOp>(loc, rhsZero, lhsInf));
				auto canonicalNaN = builder.create<mlir::arith::ConstantIntOp>(loc, 0x7FC00000u, 32);
				resultBits = builder.create<mlir::arith::SelectOp>(loc, special, canonicalNaN, resultBits);
			}
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest, resultBits);
			break;
		}
		case IROp::FSqrt: {
			auto resultBits = builder.create<mlir::func::CallOp>(loc, floatSqrt,
				mlir::ValueRange{LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.src1)}).getResult(0);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				resultBits);
			break;
		}
		case IROp::FRound:
		case IROp::FTrunc:
		case IROp::FCeil:
		case IROp::FFloor:
		case IROp::FCvtWS: {
			mlir::Value mode;
			if (inst.op == IROp::FCvtWS) {
				mode = builder.create<mlir::arith::AndIOp>(loc,
					LoadReg(builder, loc, state, MIPSStateLayout::FCR31),
					builder.create<mlir::arith::ConstantIntOp>(loc, 3, 32));
			} else {
				uint32_t fixedMode = 0;
				if (inst.op == IROp::FTrunc) fixedMode = 1;
				if (inst.op == IROp::FCeil) fixedMode = 2;
				if (inst.op == IROp::FFloor) fixedMode = 3;
				mode = builder.create<mlir::arith::ConstantIntOp>(loc, fixedMode, 32);
			}
			auto converted = builder.create<mlir::func::CallOp>(loc, floatToInt,
				mlir::ValueRange{LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.src1), mode}).getResult(0);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				converted);
			break;
		}
		case IROp::FCvtSW:
		case IROp::OptFCvtSWFromGPR: {
			const auto source = inst.op == IROp::FCvtSW
				? MIPSStateLayout::FPRBase + inst.src1 : inst.src1;
			auto converted = builder.create<mlir::arith::SIToFPOp>(loc,
				builder.getF32Type(), LoadReg(builder, loc, state, source));
			auto resultBits = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getI32Type(), converted);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				resultBits);
			break;
		}
		case IROp::FCvtScaledSW: {
			auto converted = builder.create<mlir::arith::SIToFPOp>(loc,
				builder.getF32Type(), LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.src1));
			auto scaleBits = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x3F800000u - ((inst.src2 & 31u) << 23), 32);
			auto scale = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), scaleBits);
			auto scaled = builder.create<mlir::arith::MulFOp>(loc, converted, scale);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				builder.create<mlir::arith::BitcastOp>(loc, builder.getI32Type(), scaled));
			break;
		}
		case IROp::FCvtScaledWS: {
			auto converted = builder.create<mlir::func::CallOp>(loc, floatToScaledInt,
				mlir::ValueRange{
					LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1),
					builder.create<mlir::arith::ConstantIntOp>(loc, inst.src2 & 31u, 32),
					builder.create<mlir::arith::ConstantIntOp>(loc, inst.src2 >> 6, 32),
				}).getResult(0);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				converted);
			break;
		}
		case IROp::FSin:
		case IROp::FCos:
		case IROp::FRSqrt:
		case IROp::FRecip:
		case IROp::FAsin: {
			uint32_t operation = 0;
			if (inst.op == IROp::FCos) operation = 1;
			if (inst.op == IROp::FRSqrt) operation = 2;
			if (inst.op == IROp::FRecip) operation = 3;
			if (inst.op == IROp::FAsin) operation = 4;
			auto resultBits = builder.create<mlir::func::CallOp>(loc, floatUnary,
				mlir::ValueRange{
					LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1),
					builder.create<mlir::arith::ConstantIntOp>(loc, operation, 32),
				}).getResult(0);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				resultBits);
			break;
		}
		case IROp::FMin:
		case IROp::FMax: {
			auto lhsBits = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			auto rhsBits = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src2);
			auto lhs = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), lhsBits);
			auto rhs = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), rhsBits);

			auto magnitudeMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x7FFFFFFFu, 32);
			auto infinity = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x7F800000u, 32);
			auto lhsMagnitude = builder.create<mlir::arith::AndIOp>(loc,
				lhsBits, magnitudeMask);
			auto rhsMagnitude = builder.create<mlir::arith::AndIOp>(loc,
				rhsBits, magnitudeMask);
			auto lhsNaN = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::ugt, lhsMagnitude, infinity);
			auto rhsNaN = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::ugt, rhsMagnitude, infinity);
			auto anyNaN = builder.create<mlir::arith::OrIOp>(loc, lhsNaN, rhsNaN);

			auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			auto lhsNegative = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::slt, lhsBits, zero);
			auto rhsNegative = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::slt, rhsBits, zero);
			auto bothNegative = builder.create<mlir::arith::AndIOp>(loc,
				lhsNegative, rhsNegative);
			auto lhsSignedLess = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::slt, lhsBits, rhsBits);
			auto signedMin = builder.create<mlir::arith::SelectOp>(loc,
				lhsSignedLess, lhsBits, rhsBits);
			auto signedMax = builder.create<mlir::arith::SelectOp>(loc,
				lhsSignedLess, rhsBits, lhsBits);
			mlir::Value nanResult;
			if (inst.op == IROp::FMin)
				nanResult = builder.create<mlir::arith::SelectOp>(loc,
					bothNegative, signedMax, signedMin);
			else
				nanResult = builder.create<mlir::arith::SelectOp>(loc,
					bothNegative, signedMin, signedMax);

			mlir::Value normalResult;
			if (inst.op == IROp::FMin) {
				auto rhsLess = builder.create<mlir::arith::CmpFOp>(loc,
					mlir::arith::CmpFPredicate::OLT, rhs, lhs);
				normalResult = builder.create<mlir::arith::SelectOp>(loc,
					rhsLess, rhsBits, lhsBits);
			} else {
				auto lhsLess = builder.create<mlir::arith::CmpFOp>(loc,
					mlir::arith::CmpFPredicate::OLT, lhs, rhs);
				normalResult = builder.create<mlir::arith::SelectOp>(loc,
					lhsLess, rhsBits, lhsBits);
			}
			auto resultBits = builder.create<mlir::arith::SelectOp>(loc,
				anyNaN, nanResult, normalResult);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				resultBits);
			break;
		}
		case IROp::FCmp: {
			auto lhs = builder.create<mlir::arith::BitcastOp>(loc, builder.getF32Type(),
				LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1));
			auto rhs = builder.create<mlir::arith::BitcastOp>(loc, builder.getF32Type(),
				LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src2));
			mlir::Value compared;
			switch (inst.dest) {
			case IRFpCompareMode::False:
				compared = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1); break;
			case IRFpCompareMode::EitherUnordered:
				compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNO, lhs, rhs); break;
			case IRFpCompareMode::EqualOrdered:
				compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs); break;
			case IRFpCompareMode::EqualUnordered:
				compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UEQ, lhs, rhs); break;
			case IRFpCompareMode::LessOrdered:
				compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs); break;
			case IRFpCompareMode::LessUnordered:
				compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ULT, lhs, rhs); break;
			case IRFpCompareMode::LessEqualOrdered:
				compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs); break;
			case IRFpCompareMode::LessEqualUnordered:
				compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ULE, lhs, rhs); break;
			default:
				compared = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1); break;
			}
			StoreReg(builder, loc, state, MIPSStateLayout::FPCond,
				builder.create<mlir::arith::ExtUIOp>(loc, builder.getI32Type(), compared));
			break;
		}
		case IROp::FCmovVfpuCC: {
			auto cc = LoadReg(builder, loc, state,
				MIPSStateLayout::VFPUCtrlBase + 3);
			auto bit = builder.create<mlir::arith::ConstantIntOp>(loc,
				inst.src2 & 0x0F, 32);
			auto selected = builder.create<mlir::arith::AndIOp>(loc,
				builder.create<mlir::arith::ShRUIOp>(loc, cc, bit),
				builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32));
			auto wanted = builder.create<mlir::arith::ConstantIntOp>(loc,
				inst.src2 >> 7, 32);
			auto move = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::eq, selected, wanted);
			auto oldValue = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.dest);
			auto newValue = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				builder.create<mlir::arith::SelectOp>(loc, move, newValue, oldValue));
			break;
		}
		case IROp::FCmpVfpuBit: {
			auto lhsBits = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			auto rhsBits = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src2);
			auto lhs = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), lhsBits);
			auto rhs = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), rhsBits);
			auto zeroBits = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			auto zeroFloat = builder.create<mlir::arith::BitcastOp>(loc,
				builder.getF32Type(), zeroBits);
			auto magnitudeMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x7FFFFFFFu, 32);
			auto infinity = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x7F800000u, 32);
			auto lhsMagnitude = builder.create<mlir::arith::AndIOp>(loc,
				lhsBits, magnitudeMask);
			auto isNaN = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::ugt, lhsMagnitude, infinity);
			auto isInf = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::eq, lhsMagnitude, infinity);
			auto isSpecial = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::uge, lhsMagnitude, infinity);
			mlir::Value compared;
			switch (inst.dest & 0x0F) {
			case 0: compared = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1); break;
			case 1: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs); break;
			case 2: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs); break;
			case 3: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs); break;
			case 4: compared = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1); break;
			case 5: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, lhs, rhs); break;
			case 6: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs); break;
			case 7: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs); break;
			case 8: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, lhs, zeroFloat); break;
			case 9: compared = isNaN; break;
			case 10: compared = isInf; break;
			case 11: compared = isSpecial; break;
			case 12: compared = builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::UNE, lhs, zeroFloat); break;
			case 13: compared = builder.create<mlir::arith::XOrIOp>(loc, isNaN, builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1)); break;
			case 14: compared = builder.create<mlir::arith::XOrIOp>(loc, isInf, builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1)); break;
			default: compared = builder.create<mlir::arith::XOrIOp>(loc, isSpecial, builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1)); break;
			}
			auto cc = LoadReg(builder, loc, state,
				MIPSStateLayout::VFPUCtrlBase + 3);
			auto bitMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				1u << (inst.dest >> 4), 32);
			auto clearMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				~(1u << (inst.dest >> 4)), 32);
			auto cleared = builder.create<mlir::arith::AndIOp>(loc, cc, clearMask);
			auto resultBit = builder.create<mlir::arith::SelectOp>(loc, compared,
				bitMask, builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32));
			StoreReg(builder, loc, state, MIPSStateLayout::VFPUCtrlBase + 3,
				builder.create<mlir::arith::OrIOp>(loc, cleared, resultBit));
			break;
		}
		case IROp::FCmpVfpuAggregate: {
			auto cc = LoadReg(builder, loc, state,
				MIPSStateLayout::VFPUCtrlBase + 3);
			auto mask = builder.create<mlir::arith::ConstantIntOp>(loc, inst.dest, 32);
			auto masked = builder.create<mlir::arith::AndIOp>(loc, cc, mask);
			auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			auto any = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::ne, masked, zero);
			auto all = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::eq, masked, mask);
			auto anyBit = builder.create<mlir::arith::SelectOp>(loc, any,
				builder.create<mlir::arith::ConstantIntOp>(loc, 0x10, 32), zero);
			auto allBit = builder.create<mlir::arith::SelectOp>(loc, all,
				builder.create<mlir::arith::ConstantIntOp>(loc, 0x20, 32), zero);
			auto cleared = builder.create<mlir::arith::AndIOp>(loc, cc,
				builder.create<mlir::arith::ConstantIntOp>(loc, ~0x30u, 32));
			StoreReg(builder, loc, state, MIPSStateLayout::VFPUCtrlBase + 3,
				builder.create<mlir::arith::OrIOp>(loc, cleared,
					builder.create<mlir::arith::OrIOp>(loc, anyBit, allBit)));
			break;
		}
		case IROp::FMovFromGPR:
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				LoadReg(builder, loc, state, inst.src1));
			break;
		case IROp::FMovToGPR:
			StoreReg(builder, loc, state, inst.dest,
				LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1));
			break;
		case IROp::OptFMovToGPRShr8: {
			auto amount = builder.create<mlir::arith::ConstantIntOp>(loc, 8, 32);
			StoreReg(builder, loc, state, inst.dest,
				builder.create<mlir::arith::ShRUIOp>(loc,
					LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1), amount));
			break;
		}
		case IROp::FpCondFromReg:
			StoreReg(builder, loc, state, MIPSStateLayout::FPCond,
				LoadReg(builder, loc, state, inst.dest));
			break;
		case IROp::FpCondToReg:
			StoreReg(builder, loc, state, inst.dest,
				LoadReg(builder, loc, state, MIPSStateLayout::FPCond));
			break;
		case IROp::FpCtrlFromReg: {
			auto mask = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x0181FFFFu, 32);
			mlir::Value control = builder.create<mlir::arith::AndIOp>(loc,
				LoadReg(builder, loc, state, inst.src1), mask);
			StoreReg(builder, loc, state, MIPSStateLayout::FCR31, control);
			auto shift = builder.create<mlir::arith::ConstantIntOp>(loc, 23, 32);
			auto one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
			mlir::Value condition = builder.create<mlir::arith::AndIOp>(loc,
				builder.create<mlir::arith::ShRUIOp>(loc, control, shift), one);
			StoreReg(builder, loc, state, MIPSStateLayout::FPCond, condition);
			break;
		}
		case IROp::FpCtrlToReg: {
			auto conditionMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				1u << 23, 32);
			auto clearMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				~(1u << 23), 32);
			mlir::Value control = builder.create<mlir::arith::AndIOp>(loc,
				LoadReg(builder, loc, state, MIPSStateLayout::FCR31), clearMask);
			mlir::Value condition = builder.create<mlir::arith::AndIOp>(loc,
				LoadReg(builder, loc, state, MIPSStateLayout::FPCond),
				builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32));
			auto shift = builder.create<mlir::arith::ConstantIntOp>(loc, 23, 32);
			condition = builder.create<mlir::arith::ShLIOp>(loc, condition, shift);
			control = builder.create<mlir::arith::OrIOp>(loc, control,
				builder.create<mlir::arith::AndIOp>(loc, condition, conditionMask));
			StoreReg(builder, loc, state, MIPSStateLayout::FCR31, control);
			StoreReg(builder, loc, state, inst.dest, control);
			break;
		}
		case IROp::VfpuCtrlToReg:
			StoreReg(builder, loc, state, inst.dest, LoadReg(builder, loc, state,
				MIPSStateLayout::VFPUCtrlBase + inst.src1));
			break;
		case IROp::SetCtrlVFPU:
			StoreReg(builder, loc, state, MIPSStateLayout::VFPUCtrlBase + inst.dest,
				builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32));
			break;
		case IROp::SetCtrlVFPUReg:
			StoreReg(builder, loc, state, MIPSStateLayout::VFPUCtrlBase + inst.dest,
				LoadReg(builder, loc, state, inst.src1));
			break;
		case IROp::SetCtrlVFPUFReg:
			StoreReg(builder, loc, state, MIPSStateLayout::VFPUCtrlBase + inst.dest,
				LoadReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.src1));
			break;
		case IROp::Vec4Init: {
			for (unsigned lane = 0; lane < 4; ++lane) {
				uint32_t bits = 0;
				if (inst.src1 == static_cast<IRReg>(Vec4Init::AllONE) ||
					inst.src1 == static_cast<IRReg>(Vec4Init::Set_1000) + lane)
					bits = 0x3F800000u;
				else if (inst.src1 == static_cast<IRReg>(Vec4Init::AllMinusONE))
					bits = 0xBF800000u;
				StoreReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.dest + lane,
					builder.create<mlir::arith::ConstantIntOp>(loc, bits, 32));
			}
			break;
		}
		case IROp::Vec4Shuffle: {
			mlir::Value lanes[4];
			for (unsigned lane = 0; lane < 4; ++lane) {
				unsigned sourceLane = (inst.src2 >> (lane * 2)) & 3;
				lanes[lane] = LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.src1 + sourceLane);
			}
			for (unsigned lane = 0; lane < 4; ++lane)
				StoreReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.dest + lane, lanes[lane]);
			break;
		}
		case IROp::Vec4Blend: {
			mlir::Value lanes[4];
			for (unsigned lane = 0; lane < 4; ++lane) {
				auto source = ((inst.constant >> lane) & 1) ? inst.src2 : inst.src1;
				lanes[lane] = LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + source + lane);
			}
			for (unsigned lane = 0; lane < 4; ++lane)
				StoreReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.dest + lane, lanes[lane]);
			break;
		}
		case IROp::Vec4Mov:
		case IROp::Vec4Neg:
		case IROp::Vec4Abs: {
			mlir::Value lanes[4];
			auto signMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x80000000u, 32);
			auto magnitudeMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				0x7FFFFFFFu, 32);
			for (unsigned lane = 0; lane < 4; ++lane) {
				lanes[lane] = LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.src1 + lane);
				if (inst.op == IROp::Vec4Neg)
					lanes[lane] = builder.create<mlir::arith::XOrIOp>(loc,
						lanes[lane], signMask);
				if (inst.op == IROp::Vec4Abs)
					lanes[lane] = builder.create<mlir::arith::AndIOp>(loc,
						lanes[lane], magnitudeMask);
			}
			for (unsigned lane = 0; lane < 4; ++lane)
				StoreReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.dest + lane, lanes[lane]);
			break;
		}
		case IROp::Vec4Add:
		case IROp::Vec4Sub:
		case IROp::Vec4Mul:
		case IROp::Vec4Div:
		case IROp::Vec4Scale: {
			mlir::Value results[4];
			mlir::Value scale;
			if (inst.op == IROp::Vec4Scale)
				scale = builder.create<mlir::arith::BitcastOp>(loc,
					builder.getF32Type(), LoadReg(builder, loc, state,
						MIPSStateLayout::FPRBase + inst.src2));
			for (unsigned lane = 0; lane < 4; ++lane) {
				auto lhs = builder.create<mlir::arith::BitcastOp>(loc,
					builder.getF32Type(), LoadReg(builder, loc, state,
						MIPSStateLayout::FPRBase + inst.src1 + lane));
				mlir::Value rhs = scale;
				if (inst.op != IROp::Vec4Scale)
					rhs = builder.create<mlir::arith::BitcastOp>(loc,
						builder.getF32Type(), LoadReg(builder, loc, state,
							MIPSStateLayout::FPRBase + inst.src2 + lane));
				mlir::Value floatResult;
				if (inst.op == IROp::Vec4Add)
					floatResult = builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
				if (inst.op == IROp::Vec4Sub)
					floatResult = builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
				if (inst.op == IROp::Vec4Mul || inst.op == IROp::Vec4Scale)
					floatResult = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
				if (inst.op == IROp::Vec4Div)
					floatResult = builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
				results[lane] = builder.create<mlir::arith::BitcastOp>(loc,
					builder.getI32Type(), floatResult);
			}
			for (unsigned lane = 0; lane < 4; ++lane)
				StoreReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.dest + lane, results[lane]);
			break;
		}
		case IROp::Vec4Dot: {
			mlir::Value products[4];
			for (unsigned lane = 0; lane < 4; ++lane) {
				auto lhs = builder.create<mlir::arith::BitcastOp>(loc,
					builder.getF32Type(), LoadReg(builder, loc, state,
						MIPSStateLayout::FPRBase + inst.src1 + lane));
				auto rhs = builder.create<mlir::arith::BitcastOp>(loc,
					builder.getF32Type(), LoadReg(builder, loc, state,
						MIPSStateLayout::FPRBase + inst.src2 + lane));
				products[lane] = builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
			}
			auto sum = builder.create<mlir::arith::AddFOp>(loc, products[0], products[1]);
			sum = builder.create<mlir::arith::AddFOp>(loc, sum, products[2]);
			sum = builder.create<mlir::arith::AddFOp>(loc, sum, products[3]);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				builder.create<mlir::arith::BitcastOp>(loc, builder.getI32Type(), sum));
			break;
		}
		case IROp::Vec2Unpack16To31:
		case IROp::Vec2Unpack16To32: {
			auto input = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			auto shift16 = builder.create<mlir::arith::ConstantIntOp>(loc, 16, 32);
			mlir::Value low = builder.create<mlir::arith::ShLIOp>(loc, input, shift16);
			mlir::Value high = builder.create<mlir::arith::AndIOp>(loc, input,
				builder.create<mlir::arith::ConstantIntOp>(loc, 0xFFFF0000u, 32));
			if (inst.op == IROp::Vec2Unpack16To31) {
				auto one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
				low = builder.create<mlir::arith::ShRUIOp>(loc, low, one);
				high = builder.create<mlir::arith::ShRUIOp>(loc, high, one);
			}
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest, low);
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest + 1, high);
			break;
		}
		case IROp::Vec4Unpack8To32: {
			auto input = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			for (unsigned lane = 0; lane < 4; ++lane) {
				auto right = builder.create<mlir::arith::ConstantIntOp>(loc, lane * 8, 32);
				mlir::Value byte = builder.create<mlir::arith::AndIOp>(loc,
					builder.create<mlir::arith::ShRUIOp>(loc, input, right),
					builder.create<mlir::arith::ConstantIntOp>(loc, 0xFF, 32));
				auto expanded = builder.create<mlir::arith::ShLIOp>(loc, byte,
					builder.create<mlir::arith::ConstantIntOp>(loc, 24, 32));
				StoreReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.dest + lane, expanded);
			}
			break;
		}
		case IROp::Vec4DuplicateUpperBitsAndShift1:
		case IROp::Vec4ClampToZero:
		case IROp::Vec2ClampToZero: {
			unsigned lanes = inst.op == IROp::Vec2ClampToZero ? 2 : 4;
			mlir::Value results[4];
			for (unsigned lane = 0; lane < lanes; ++lane) {
				auto input = LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.src1 + lane);
				if (inst.op == IROp::Vec4DuplicateUpperBitsAndShift1) {
					auto by8 = builder.create<mlir::arith::ConstantIntOp>(loc, 8, 32);
					auto by16 = builder.create<mlir::arith::ConstantIntOp>(loc, 16, 32);
					input = builder.create<mlir::arith::OrIOp>(loc, input,
						builder.create<mlir::arith::ShRUIOp>(loc, input, by8));
					input = builder.create<mlir::arith::OrIOp>(loc, input,
						builder.create<mlir::arith::ShRUIOp>(loc, input, by16));
					results[lane] = builder.create<mlir::arith::ShRUIOp>(loc, input,
						builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32));
				} else {
					auto nonnegative = builder.create<mlir::arith::CmpIOp>(loc,
						mlir::arith::CmpIPredicate::sge, input,
						builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32));
					results[lane] = builder.create<mlir::arith::SelectOp>(loc,
						nonnegative, input,
						builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32));
				}
			}
			for (unsigned lane = 0; lane < lanes; ++lane)
				StoreReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.dest + lane, results[lane]);
			break;
		}
		case IROp::Vec2Pack31To16:
		case IROp::Vec2Pack32To16: {
			auto first = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1);
			auto second = LoadReg(builder, loc, state,
				MIPSStateLayout::FPRBase + inst.src1 + 1);
			unsigned shift = inst.op == IROp::Vec2Pack31To16 ? 15 : 16;
			auto low = builder.create<mlir::arith::AndIOp>(loc,
				builder.create<mlir::arith::ShRUIOp>(loc, first,
					builder.create<mlir::arith::ConstantIntOp>(loc, shift, 32)),
				builder.create<mlir::arith::ConstantIntOp>(loc, 0xFFFF, 32));
			mlir::Value high = second;
			if (shift == 15)
				high = builder.create<mlir::arith::ShLIOp>(loc, high,
					builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32));
			high = builder.create<mlir::arith::AndIOp>(loc, high,
				builder.create<mlir::arith::ConstantIntOp>(loc, 0xFFFF0000u, 32));
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest,
				builder.create<mlir::arith::OrIOp>(loc, low, high));
			break;
		}
		case IROp::Vec4Pack31To8:
		case IROp::Vec4Pack32To8: {
			mlir::Value packed = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			unsigned sourceShift = inst.op == IROp::Vec4Pack31To8 ? 23 : 24;
			for (unsigned lane = 0; lane < 4; ++lane) {
				auto source = LoadReg(builder, loc, state,
					MIPSStateLayout::FPRBase + inst.src1 + lane);
				mlir::Value byte = builder.create<mlir::arith::AndIOp>(loc,
					builder.create<mlir::arith::ShRUIOp>(loc, source,
						builder.create<mlir::arith::ConstantIntOp>(loc, sourceShift, 32)),
					builder.create<mlir::arith::ConstantIntOp>(loc, 0xFF, 32));
				byte = builder.create<mlir::arith::ShLIOp>(loc, byte,
					builder.create<mlir::arith::ConstantIntOp>(loc, lane * 8, 32));
				packed = builder.create<mlir::arith::OrIOp>(loc, packed, byte);
			}
			StoreReg(builder, loc, state, MIPSStateLayout::FPRBase + inst.dest, packed);
			break;
		}
		case IROp::Add:
		case IROp::Sub:
		case IROp::And:
		case IROp::Or:
		case IROp::Xor: {
			auto lhs = LoadReg(builder, loc, state, inst.src1);
			auto rhs = LoadReg(builder, loc, state, inst.src2);
			if (inst.op == IROp::Add) value = builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
			if (inst.op == IROp::Sub) value = builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
			if (inst.op == IROp::And) value = builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
			if (inst.op == IROp::Or) value = builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
			if (inst.op == IROp::Xor) value = builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::AddConst:
		case IROp::OptAddConst:
		case IROp::SubConst:
		case IROp::AndConst:
		case IROp::OrConst:
		case IROp::XorConst:
		case IROp::OptAndConst:
		case IROp::OptOrConst: {
			const IRReg source = (inst.op == IROp::OptAddConst ||
				inst.op == IROp::OptAndConst || inst.op == IROp::OptOrConst) ?
				inst.dest : inst.src1;
			auto lhs = LoadReg(builder, loc, state, source);
			auto rhs = builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32);
			if (inst.op == IROp::AddConst || inst.op == IROp::OptAddConst) value = builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
			if (inst.op == IROp::SubConst) value = builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
			if (inst.op == IROp::AndConst || inst.op == IROp::OptAndConst) value = builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
			if (inst.op == IROp::OrConst || inst.op == IROp::OptOrConst) value = builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
			if (inst.op == IROp::XorConst) value = builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::Shl:
		case IROp::Shr:
		case IROp::Sar: {
			auto input = LoadReg(builder, loc, state, inst.src1);
			auto amount = LoadReg(builder, loc, state, inst.src2);
			auto mask = builder.create<mlir::arith::ConstantIntOp>(loc, 31, 32);
			amount = builder.create<mlir::arith::AndIOp>(loc, amount, mask);
			if (inst.op == IROp::Shl) value = builder.create<mlir::arith::ShLIOp>(loc, input, amount);
			if (inst.op == IROp::Shr) value = builder.create<mlir::arith::ShRUIOp>(loc, input, amount);
			if (inst.op == IROp::Sar) value = builder.create<mlir::arith::ShRSIOp>(loc, input, amount);
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::ShlImm:
		case IROp::ShrImm:
		case IROp::SarImm: {
			auto input = LoadReg(builder, loc, state, inst.src1);
			auto amount = builder.create<mlir::arith::ConstantIntOp>(loc,
				inst.src2 & 31, 32);
			if (inst.op == IROp::ShlImm) value = builder.create<mlir::arith::ShLIOp>(loc, input, amount);
			if (inst.op == IROp::ShrImm) value = builder.create<mlir::arith::ShRUIOp>(loc, input, amount);
			if (inst.op == IROp::SarImm) value = builder.create<mlir::arith::ShRSIOp>(loc, input, amount);
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::Slt:
		case IROp::SltU:
		case IROp::SltConst:
		case IROp::SltUConst: {
			auto lhs = LoadReg(builder, loc, state, inst.src1);
			mlir::Value rhs = (inst.op == IROp::SltConst || inst.op == IROp::SltUConst) ?
				mlir::Value(builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32)) :
				LoadReg(builder, loc, state, inst.src2);
			auto predicate = (inst.op == IROp::SltU || inst.op == IROp::SltUConst) ?
				mlir::arith::CmpIPredicate::ult : mlir::arith::CmpIPredicate::slt;
			auto compared = builder.create<mlir::arith::CmpIOp>(loc, predicate, lhs, rhs);
			value = builder.create<mlir::arith::ExtUIOp>(loc, builder.getI32Type(), compared);
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::Load32:
		case IROp::Store32:
		case IROp::LoadFloat:
		case IROp::StoreFloat: {
			auto effectiveAddress = builder.create<mlir::arith::AddIOp>(loc,
				LoadReg(builder, loc, state, inst.src1),
				builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32));
			if (inst.op == IROp::Load32 || inst.op == IROp::LoadFloat) {
				auto size = builder.create<mlir::arith::ConstantIntOp>(loc, 4, 32);
				auto sign = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
				value = builder.create<mlir::func::CallOp>(loc, readMemory,
					mlir::ValueRange{effectiveAddress, size, sign}).getResult(0);
				const auto destination = inst.op == IROp::LoadFloat
					? MIPSStateLayout::FPRBase + inst.dest : inst.dest;
				StoreReg(builder, loc, state, destination, value);
			} else {
				auto size = builder.create<mlir::arith::ConstantIntOp>(loc, 4, 32);
				const auto source = inst.op == IROp::StoreFloat
					? MIPSStateLayout::FPRBase + inst.src3 : inst.src3;
				builder.create<mlir::func::CallOp>(loc, writeMemory,
					mlir::ValueRange{effectiveAddress, size,
						LoadReg(builder, loc, state, source)});
			}
			break;
		}
		case IROp::LoadVec4:
		case IROp::StoreVec4: {
			auto base = builder.create<mlir::arith::AddIOp>(loc,
				LoadReg(builder, loc, state, inst.src1),
				builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32));
			auto size = builder.create<mlir::arith::ConstantIntOp>(loc, 4, 32);
			auto sign = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			if (inst.op == IROp::LoadVec4) {
				mlir::Value lanes[4];
				for (unsigned lane = 0; lane < 4; ++lane) {
					auto address = builder.create<mlir::arith::AddIOp>(loc, base,
						builder.create<mlir::arith::ConstantIntOp>(loc, lane * 4, 32));
					lanes[lane] = builder.create<mlir::func::CallOp>(loc, readMemory,
						mlir::ValueRange{address, size, sign}).getResult(0);
				}
				for (unsigned lane = 0; lane < 4; ++lane)
					StoreReg(builder, loc, state,
						MIPSStateLayout::FPRBase + inst.dest + lane, lanes[lane]);
			} else {
				for (unsigned lane = 0; lane < 4; ++lane) {
					auto address = builder.create<mlir::arith::AddIOp>(loc, base,
						builder.create<mlir::arith::ConstantIntOp>(loc, lane * 4, 32));
					builder.create<mlir::func::CallOp>(loc, writeMemory,
						mlir::ValueRange{address, size, LoadReg(builder, loc, state,
							MIPSStateLayout::FPRBase + inst.dest + lane)});
				}
			}
			break;
		}
		case IROp::Load32Left:
		case IROp::Load32Right:
		case IROp::Store32Left:
		case IROp::Store32Right: {
			auto address = builder.create<mlir::arith::AddIOp>(loc,
				LoadReg(builder, loc, state, inst.src1),
				builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32));
			auto threeMask = builder.create<mlir::arith::ConstantIntOp>(loc, 3, 32);
			auto byteOffset = builder.create<mlir::arith::AndIOp>(loc, address, threeMask);
			auto three = builder.create<mlir::arith::ConstantIntOp>(loc, 3, 32);
			auto shift = builder.create<mlir::arith::ShLIOp>(loc, byteOffset, three);
			auto twentyFour = builder.create<mlir::arith::ConstantIntOp>(loc, 24, 32);
			auto inverseShift = builder.create<mlir::arith::SubIOp>(loc, twentyFour, shift);
			auto alignmentMask = builder.create<mlir::arith::ConstantIntOp>(loc,
				0xFFFFFFFCu, 32);
			auto aligned = builder.create<mlir::arith::AndIOp>(loc, address, alignmentMask);
			auto size = builder.create<mlir::arith::ConstantIntOp>(loc, 4, 32);
			auto sign = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			auto memoryWord = builder.create<mlir::func::CallOp>(loc, readMemory,
				mlir::ValueRange{aligned, size, sign}).getResult(0);
			if (inst.op == IROp::Load32Left || inst.op == IROp::Load32Right) {
				auto oldDest = LoadReg(builder, loc, state, inst.dest);
				uint32_t maskValue = inst.op == IROp::Load32Left ? 0x00FFFFFF : 0xFFFFFF00;
				auto mask = builder.create<mlir::arith::ConstantIntOp>(loc, maskValue, 32);
				mlir::Value destMask = inst.op == IROp::Load32Left ?
					mlir::Value(builder.create<mlir::arith::ShRUIOp>(loc, mask, shift)) :
					mlir::Value(builder.create<mlir::arith::ShLIOp>(loc, mask, inverseShift));
				mlir::Value loaded = inst.op == IROp::Load32Left ?
					mlir::Value(builder.create<mlir::arith::ShLIOp>(loc, memoryWord, inverseShift)) :
					mlir::Value(builder.create<mlir::arith::ShRUIOp>(loc, memoryWord, shift));
				value = builder.create<mlir::arith::OrIOp>(loc,
					builder.create<mlir::arith::AndIOp>(loc, oldDest, destMask), loaded);
				StoreReg(builder, loc, state, inst.dest, value);
			} else {
				auto source = LoadReg(builder, loc, state, inst.src3);
				uint32_t maskValue = inst.op == IROp::Store32Left ? 0xFFFFFF00 : 0x00FFFFFF;
				auto mask = builder.create<mlir::arith::ConstantIntOp>(loc, maskValue, 32);
				mlir::Value memoryMask = inst.op == IROp::Store32Left ?
					mlir::Value(builder.create<mlir::arith::ShLIOp>(loc, mask, shift)) :
					mlir::Value(builder.create<mlir::arith::ShRUIOp>(loc, mask, inverseShift));
				mlir::Value stored = inst.op == IROp::Store32Left ?
					mlir::Value(builder.create<mlir::arith::ShRUIOp>(loc, source, inverseShift)) :
					mlir::Value(builder.create<mlir::arith::ShLIOp>(loc, source, shift));
				auto combined = builder.create<mlir::arith::OrIOp>(loc, stored,
					builder.create<mlir::arith::AndIOp>(loc, memoryWord, memoryMask));
				builder.create<mlir::func::CallOp>(loc, writeMemory,
					mlir::ValueRange{aligned, size, combined});
			}
			break;
		}
		case IROp::Load32Linked:
		case IROp::Store32Conditional: {
			auto address = builder.create<mlir::arith::AddIOp>(loc,
				LoadReg(builder, loc, state, inst.src1),
				builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32));
			auto size = builder.create<mlir::arith::ConstantIntOp>(loc, 4, 32);
			auto one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
			if (inst.op == IROp::Load32Linked) {
				auto sign = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
				value = builder.create<mlir::func::CallOp>(loc, readMemory,
					mlir::ValueRange{address, size, sign}).getResult(0);
				StoreReg(builder, loc, state, inst.dest, value);
				StoreReg(builder, loc, state, kLlBitWord, one);
			} else {
				auto llBit = LoadReg(builder, loc, state, kLlBitWord);
				value = builder.create<mlir::func::CallOp>(loc, writeMemoryConditional,
					mlir::ValueRange{address, size,
						LoadReg(builder, loc, state, inst.src3), llBit}).getResult(0);
				StoreReg(builder, loc, state, inst.dest, value);
			}
			break;
		}
		case IROp::Load8:
		case IROp::Load8Ext:
		case IROp::Load16:
		case IROp::Load16Ext:
		case IROp::Store8:
		case IROp::Store16: {
			auto address = LoadReg(builder, loc, state, inst.src1);
			auto offset = builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32);
			address = builder.create<mlir::arith::AddIOp>(loc, address, offset);
			bool half = inst.op == IROp::Load16 || inst.op == IROp::Load16Ext ||
				inst.op == IROp::Store16;
			auto size = builder.create<mlir::arith::ConstantIntOp>(loc, half ? 2 : 1, 32);
			if (inst.op == IROp::Store8 || inst.op == IROp::Store16) {
				auto source = LoadReg(builder, loc, state, inst.src3);
				builder.create<mlir::func::CallOp>(loc, writeMemory,
					mlir::ValueRange{address, size, source});
			} else {
				bool signExtend = inst.op == IROp::Load8Ext || inst.op == IROp::Load16Ext;
				auto sign = builder.create<mlir::arith::ConstantIntOp>(loc,
					signExtend ? 1 : 0, 32);
				value = builder.create<mlir::func::CallOp>(loc, readMemory,
					mlir::ValueRange{address, size, sign}).getResult(0);
				StoreReg(builder, loc, state, inst.dest, value);
			}
			break;
		}
		case IROp::MtLo:
			StoreReg(builder, loc, state, kLoWord, LoadReg(builder, loc, state, inst.src1));
			break;
		case IROp::MtHi:
			StoreReg(builder, loc, state, kHiWord, LoadReg(builder, loc, state, inst.src1));
			break;
		case IROp::MfLo:
			StoreReg(builder, loc, state, inst.dest, LoadReg(builder, loc, state, kLoWord));
			break;
		case IROp::MfHi:
			StoreReg(builder, loc, state, inst.dest, LoadReg(builder, loc, state, kHiWord));
			break;
		case IROp::Mult:
		case IROp::MultU:
		case IROp::Madd:
		case IROp::MaddU:
		case IROp::Msub:
		case IROp::MsubU: {
			auto i64 = builder.getI64Type();
			auto lhs32 = LoadReg(builder, loc, state, inst.src1);
			auto rhs32 = LoadReg(builder, loc, state, inst.src2);
			bool signedMultiply = inst.op == IROp::Mult || inst.op == IROp::Madd ||
				inst.op == IROp::Msub;
			mlir::Value lhs = signedMultiply ?
				mlir::Value(builder.create<mlir::arith::ExtSIOp>(loc, i64, lhs32)) :
				mlir::Value(builder.create<mlir::arith::ExtUIOp>(loc, i64, lhs32));
			mlir::Value rhs = signedMultiply ?
				mlir::Value(builder.create<mlir::arith::ExtSIOp>(loc, i64, rhs32)) :
				mlir::Value(builder.create<mlir::arith::ExtUIOp>(loc, i64, rhs32));
			auto product = builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
			mlir::Value result64 = product;
			if (inst.op == IROp::Madd || inst.op == IROp::MaddU ||
					inst.op == IROp::Msub || inst.op == IROp::MsubU) {
				auto lo = builder.create<mlir::arith::ExtUIOp>(loc, i64,
					LoadReg(builder, loc, state, kLoWord));
				auto hi = builder.create<mlir::arith::ExtUIOp>(loc, i64,
					LoadReg(builder, loc, state, kHiWord));
				auto shift32 = builder.create<mlir::arith::ConstantIntOp>(loc, 32, 64);
				auto accumulator = builder.create<mlir::arith::OrIOp>(loc, lo,
					builder.create<mlir::arith::ShLIOp>(loc, hi, shift32));
				bool subtract = inst.op == IROp::Msub || inst.op == IROp::MsubU;
				result64 = subtract ? mlir::Value(builder.create<mlir::arith::SubIOp>(
					loc, accumulator, product)) :
					mlir::Value(builder.create<mlir::arith::AddIOp>(loc, accumulator, product));
			}
			auto lo = builder.create<mlir::arith::TruncIOp>(loc, builder.getI32Type(), result64);
			auto shift32 = builder.create<mlir::arith::ConstantIntOp>(loc, 32, 64);
			auto high64 = builder.create<mlir::arith::ShRUIOp>(loc, result64, shift32);
			auto hi = builder.create<mlir::arith::TruncIOp>(loc, builder.getI32Type(), high64);
			StoreReg(builder, loc, state, kLoWord, lo);
			StoreReg(builder, loc, state, kHiWord, hi);
			break;
		}
		case IROp::Div:
		case IROp::DivU: {
			auto numerator = LoadReg(builder, loc, state, inst.src1);
			auto denominator = LoadReg(builder, loc, state, inst.src2);
			auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			auto one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 32);
			auto denominatorIsZero = builder.create<mlir::arith::CmpIOp>(loc,
				mlir::arith::CmpIPredicate::eq, denominator, zero);
			auto safeDenominator = builder.create<mlir::arith::SelectOp>(loc,
				denominatorIsZero, one, denominator);
			mlir::Value quotient;
			mlir::Value remainder;
			if (inst.op == IROp::Div) {
				auto i64 = builder.getI64Type();
				auto numerator64 = builder.create<mlir::arith::ExtSIOp>(loc, i64, numerator);
				auto denominator64 = builder.create<mlir::arith::ExtSIOp>(loc, i64, safeDenominator);
				quotient = builder.create<mlir::arith::TruncIOp>(loc, builder.getI32Type(),
					builder.create<mlir::arith::DivSIOp>(loc, numerator64, denominator64));
				remainder = builder.create<mlir::arith::TruncIOp>(loc, builder.getI32Type(),
					builder.create<mlir::arith::RemSIOp>(loc, numerator64, denominator64));
				auto negative = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::slt, numerator, zero);
				auto minusOne = builder.create<mlir::arith::ConstantIntOp>(loc, 0xFFFFFFFFu, 32);
				auto zeroQuotient = builder.create<mlir::arith::SelectOp>(loc,
					negative, one, minusOne);
				quotient = builder.create<mlir::arith::SelectOp>(loc, denominatorIsZero,
					zeroQuotient, quotient);
				remainder = builder.create<mlir::arith::SelectOp>(loc, denominatorIsZero,
					numerator, remainder);
				auto minInt = builder.create<mlir::arith::ConstantIntOp>(loc, 0x80000000u, 32);
				auto isMin = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::eq, numerator, minInt);
				auto isMinusOne = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::eq, denominator, minusOne);
				auto overflow = builder.create<mlir::arith::AndIOp>(loc, isMin, isMinusOne);
				remainder = builder.create<mlir::arith::SelectOp>(loc, overflow,
					minusOne, remainder);
			} else {
				quotient = builder.create<mlir::arith::DivUIOp>(loc, numerator, safeDenominator);
				remainder = builder.create<mlir::arith::RemUIOp>(loc, numerator, safeDenominator);
				auto max16 = builder.create<mlir::arith::ConstantIntOp>(loc, 0xFFFF, 32);
				auto small = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::ule, numerator, max16);
				auto minusOne = builder.create<mlir::arith::ConstantIntOp>(loc, 0xFFFFFFFFu, 32);
				auto zeroQuotient = builder.create<mlir::arith::SelectOp>(loc,
					small, max16, minusOne);
				quotient = builder.create<mlir::arith::SelectOp>(loc, denominatorIsZero,
					zeroQuotient, quotient);
				remainder = builder.create<mlir::arith::SelectOp>(loc, denominatorIsZero,
					numerator, remainder);
			}
			StoreReg(builder, loc, state, kLoWord, quotient);
			StoreReg(builder, loc, state, kHiWord, remainder);
			break;
		}
		case IROp::Neg:
		case IROp::Not: {
			auto input = LoadReg(builder, loc, state, inst.src1);
			auto constant = builder.create<mlir::arith::ConstantIntOp>(loc,
				inst.op == IROp::Neg ? 0 : 0xFFFFFFFFu, 32);
			value = inst.op == IROp::Neg ?
				mlir::Value(builder.create<mlir::arith::SubIOp>(loc, constant, input)) :
				mlir::Value(builder.create<mlir::arith::XOrIOp>(loc, input, constant));
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::Ext8to32:
		case IROp::Ext16to32: {
			auto narrowType = builder.getIntegerType(inst.op == IROp::Ext8to32 ? 8 : 16);
			auto narrow = builder.create<mlir::arith::TruncIOp>(loc, narrowType,
				LoadReg(builder, loc, state, inst.src1));
			value = builder.create<mlir::arith::ExtSIOp>(loc, builder.getI32Type(), narrow);
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::Ror:
		case IROp::RorImm: {
			auto input = LoadReg(builder, loc, state, inst.src1);
			mlir::Value amount = inst.op == IROp::Ror ?
				LoadReg(builder, loc, state, inst.src2) :
				mlir::Value(builder.create<mlir::arith::ConstantIntOp>(loc, inst.src2, 32));
			auto mask = builder.create<mlir::arith::ConstantIntOp>(loc, 31, 32);
			amount = builder.create<mlir::arith::AndIOp>(loc, amount, mask);
			auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			mlir::Value inverse = builder.create<mlir::arith::SubIOp>(loc, zero, amount);
			inverse = builder.create<mlir::arith::AndIOp>(loc, inverse, mask);
			value = builder.create<mlir::arith::OrIOp>(loc,
				builder.create<mlir::arith::ShRUIOp>(loc, input, amount),
				builder.create<mlir::arith::ShLIOp>(loc, input, inverse));
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::MovZ:
		case IROp::MovNZ: {
			auto conditionSource = LoadReg(builder, loc, state, inst.src1);
			auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			auto predicate = inst.op == IROp::MovZ ?
				mlir::arith::CmpIPredicate::eq : mlir::arith::CmpIPredicate::ne;
			auto condition = builder.create<mlir::arith::CmpIOp>(loc, predicate,
				conditionSource, zero);
			value = builder.create<mlir::arith::SelectOp>(loc, condition,
				LoadReg(builder, loc, state, inst.src2),
				LoadReg(builder, loc, state, inst.dest));
			StoreReg(builder, loc, state, inst.dest, value);
			break;
		}
		case IROp::Max:
		case IROp::Min: {
			auto lhs = LoadReg(builder, loc, state, inst.src1);
			auto rhs = LoadReg(builder, loc, state, inst.src2);
			auto predicate = inst.op == IROp::Max ?
				mlir::arith::CmpIPredicate::sgt : mlir::arith::CmpIPredicate::slt;
			auto condition = builder.create<mlir::arith::CmpIOp>(loc, predicate, lhs, rhs);
			StoreReg(builder, loc, state, inst.dest,
				builder.create<mlir::arith::SelectOp>(loc, condition, lhs, rhs));
			break;
		}
		case IROp::BSwap16:
		case IROp::BSwap32:
		case IROp::ReverseBits: {
			auto input = LoadReg(builder, loc, state, inst.src1);
			auto swapAdjacent = [&](mlir::Value x, uint32_t leftMask,
					uint32_t rightMask, unsigned shiftAmount) -> mlir::Value {
				auto lm = builder.create<mlir::arith::ConstantIntOp>(loc, leftMask, 32);
				auto rm = builder.create<mlir::arith::ConstantIntOp>(loc, rightMask, 32);
				auto shift = builder.create<mlir::arith::ConstantIntOp>(loc, shiftAmount, 32);
				return builder.create<mlir::arith::OrIOp>(loc,
					builder.create<mlir::arith::ShRUIOp>(loc,
						builder.create<mlir::arith::AndIOp>(loc, x, lm), shift),
					builder.create<mlir::arith::ShLIOp>(loc,
						builder.create<mlir::arith::AndIOp>(loc, x, rm), shift));
			};
			if (inst.op == IROp::BSwap16 || inst.op == IROp::BSwap32)
				input = swapAdjacent(input, 0xFF00FF00, 0x00FF00FF, 8);
			if (inst.op == IROp::BSwap32)
				input = swapAdjacent(input, 0xFFFF0000, 0x0000FFFF, 16);
			if (inst.op == IROp::ReverseBits) {
				input = swapAdjacent(input, 0xAAAAAAAA, 0x55555555, 1);
				input = swapAdjacent(input, 0xCCCCCCCC, 0x33333333, 2);
				input = swapAdjacent(input, 0xF0F0F0F0, 0x0F0F0F0F, 4);
				input = swapAdjacent(input, 0xFF00FF00, 0x00FF00FF, 8);
				input = swapAdjacent(input, 0xFFFF0000, 0x0000FFFF, 16);
			}
			StoreReg(builder, loc, state, inst.dest, input);
			break;
		}
		case IROp::Clz: {
			auto input = LoadReg(builder, loc, state, inst.src1);
			mlir::Value count = builder.create<mlir::arith::ConstantIntOp>(loc, 32, 32);
			auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			for (unsigned bit = 0; bit < 32; ++bit) {
				auto mask = builder.create<mlir::arith::ConstantIntOp>(loc, 1u << bit, 32);
				auto set = builder.create<mlir::arith::CmpIOp>(loc,
					mlir::arith::CmpIPredicate::ne,
					builder.create<mlir::arith::AndIOp>(loc, input, mask), zero);
				auto bitCount = builder.create<mlir::arith::ConstantIntOp>(loc, 31 - bit, 32);
				count = builder.create<mlir::arith::SelectOp>(loc, set, bitCount, count);
			}
			StoreReg(builder, loc, state, inst.dest, count);
			break;
		}
		case IROp::ExitToConst: {
			auto nextPC = builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32);
			builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{nextPC});
			emittedReturn = true;
			blockTerminated = true;
			break;
		}
		case IROp::ExitToReg: {
			builder.create<mlir::func::ReturnOp>(loc,
				mlir::ValueRange{LoadReg(builder, loc, state, inst.src1)});
			emittedReturn = true;
			blockTerminated = true;
			break;
		}
		case IROp::SetPC:
			StoreReg(builder, loc, state, MIPSStateLayout::PC,
				LoadReg(builder, loc, state, inst.src1));
			break;
		case IROp::SetPCConst:
			StoreReg(builder, loc, state, MIPSStateLayout::PC,
				builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32));
			break;
		case IROp::ExitToPC:
			builder.create<mlir::func::ReturnOp>(loc,
				mlir::ValueRange{LoadReg(builder, loc, state, MIPSStateLayout::PC)});
			emittedReturn = true;
			blockTerminated = true;
			break;
		case IROp::ExitToConstIfEq:
		case IROp::ExitToConstIfNeq:
		case IROp::ExitToConstIfGtZ:
		case IROp::ExitToConstIfGeZ:
		case IROp::ExitToConstIfLtZ:
		case IROp::ExitToConstIfLeZ: {
			auto lhs = LoadReg(builder, loc, state, inst.src1);
			mlir::Value rhs;
			mlir::arith::CmpIPredicate predicate;
			if (inst.op == IROp::ExitToConstIfEq || inst.op == IROp::ExitToConstIfNeq) {
				rhs = LoadReg(builder, loc, state, inst.src2);
				predicate = inst.op == IROp::ExitToConstIfEq ?
					mlir::arith::CmpIPredicate::eq : mlir::arith::CmpIPredicate::ne;
			} else {
				rhs = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
				if (inst.op == IROp::ExitToConstIfGtZ) predicate = mlir::arith::CmpIPredicate::sgt;
				if (inst.op == IROp::ExitToConstIfGeZ) predicate = mlir::arith::CmpIPredicate::sge;
				if (inst.op == IROp::ExitToConstIfLtZ) predicate = mlir::arith::CmpIPredicate::slt;
				if (inst.op == IROp::ExitToConstIfLeZ) predicate = mlir::arith::CmpIPredicate::sle;
			}
			auto condition = builder.create<mlir::arith::CmpIOp>(loc, predicate, lhs, rhs);
			auto *taken = builder.createBlock(&function.getBody());
			auto *continuation = builder.createBlock(&function.getBody());
			builder.setInsertionPointToEnd(entry);
			builder.create<mlir::cf::CondBranchOp>(loc, condition, taken,
				mlir::ValueRange{}, continuation, mlir::ValueRange{});
			builder.setInsertionPointToStart(taken);
			auto target = builder.create<mlir::arith::ConstantIntOp>(loc, inst.constant, 32);
			builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{target});
			entry = continuation;
			builder.setInsertionPointToStart(continuation);
			emittedReturn = true;
			break;
		}
		case IROp::ExitToConstIfFpTrue:
		case IROp::ExitToConstIfFpFalse: {
			auto fpCondition = LoadReg(builder, loc, state, MIPSStateLayout::FPCond);
			auto zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
			auto predicate = inst.op == IROp::ExitToConstIfFpTrue ?
				mlir::arith::CmpIPredicate::ne : mlir::arith::CmpIPredicate::eq;
			auto condition = builder.create<mlir::arith::CmpIOp>(loc, predicate,
				fpCondition, zero);
			auto *taken = builder.createBlock(&function.getBody());
			auto *continuation = builder.createBlock(&function.getBody());
			builder.setInsertionPointToEnd(entry);
			builder.create<mlir::cf::CondBranchOp>(loc, condition, taken,
				mlir::ValueRange{}, continuation, mlir::ValueRange{});
			builder.setInsertionPointToStart(taken);
			auto target = builder.create<mlir::arith::ConstantIntOp>(loc,
				inst.constant, 32);
			builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{target});
			entry = continuation;
			builder.setInsertionPointToStart(continuation);
			emittedReturn = true;
			break;
		}
		case IROp::Downcount: {
			auto current = LoadReg(builder, loc, state, kDowncountWord);
			auto cycles = builder.create<mlir::arith::ConstantIntOp>(loc,
				inst.constant, 32);
			StoreReg(builder, loc, state, kDowncountWord,
				builder.create<mlir::arith::SubIOp>(loc, current, cycles));
			break;
		}
		case IROp::Nop:
		case IROp::UpdateRoundingMode:
		case IROp::ValidateAddress8:
		case IROp::ValidateAddress16:
		case IROp::ValidateAddress32:
		case IROp::ValidateAddress128:
		case IROp::LogIRBlock:
			break;
		case IROp::RestoreRoundingMode:
			builder.create<mlir::func::CallOp>(loc, restoreRounding,
				mlir::ValueRange{});
			break;
		case IROp::ApplyRoundingMode:
			builder.create<mlir::func::CallOp>(loc, applyRounding,
				mlir::ValueRange{LoadReg(builder, loc, state,
					static_cast<IRReg>(MIPSStateLayout::FCR31))});
			break;
		default:
			break;
		}
	}

	if (!emittedReturn) {
		result.error = "block has no supported exit";
		return result;
	}
	std::string verifierDiagnostics;
	mlir::ScopedDiagnosticHandler diagnosticHandler(&context,
		[&](mlir::Diagnostic &diagnostic) {
			llvm::raw_string_ostream stream(verifierDiagnostics);
			diagnostic.print(stream);
			stream << '\n';
			return mlir::success();
		});
	if (mlir::failed(mlir::verify(module))) {
		result.error = "generated MLIR module failed verification";
		if (!verifierDiagnostics.empty()) {
			result.error += ": ";
			result.error += verifierDiagnostics;
		}
		return result;
	}
	result.module = module;
	result.status = CompileStatus::Compiled;
	return result;
}

}  // namespace MLIRPSP
