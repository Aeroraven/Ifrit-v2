#include "ifrit/softgraphics/engine/shadervm/spirvvec/SpvMdQuadIRGenerator.h"
#include "ifrit/softgraphics/engine/shadervm/spirvvec/SpvMdLlvmIrRepr.h"
#define SPV_ENABLE_UTILITY_CODE
#include <spirv_headers/include/spirv/unified1/spirv.hpp>

namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec {

#define DEF_PASS(x)                                                            \
  void defPass_##x(int pc, std::vector<uint32_t> params,                       \
                   SpVcVMGeneratorContext *ctx,                                \
                   SpVcQuadGroupedIRGenerator *irg)
#define CONV_PASS(x)                                                           \
  void convPass_##x(int pc, std::vector<uint32_t> params,                      \
                    SpVcVMGeneratorContext *ctx,                               \
                    SpVcQuadGroupedIRGenerator *irg)
#define GET_PARAM(x) (irg->getVariableSafe(params[(x)]))
#define GET_PARAM_CORE(x) (getVariableSafe(x))
#define GET_PARAM_SCALAR(x) (params[(x)])
#define EMIT_ERROR(...) ifritError(irg->getParsingProgress(), __VA_ARGS__);
#define EMIT_VERBOSE(...) ifritLog1(irg->getParsingProgress(), __VA_ARGS__);
#define EMIT_VERBOSE_CORE(...) ifritLog1(getParsingProgress(), __VA_ARGS__);
#define EMIT_WARN(...) ifritLog3(irg->getParsingProgress(), __VA_ARGS__);
#define EMIT_WARN_CORE(...) ifritLog3(getParsingProgress(), __VA_ARGS__);
#define EMIT_ERROR_CORE(...) ifritError(getParsingProgress(), __VA_ARGS__);

std::string readString(const std::vector<uint32_t> &data, int &pos) {
  char *str = (char *)&data[pos];
  int strlength = strlen(str);
  pos += (strlength + 3) / 4;
  return std::string(str);
}
#define GET_PARAM_STR(x) (readString(params, x))
#define GET_DEF_PASS(x) DefinitionPass::defPass_##x
#define GET_CONV_PASS(x) ConversionPass::convPass_##x

namespace IrUtil {
std::string symbolClean(std::string x) {
  // remove leading % and @
  if (x[0] == '%' || x[0] == '@') {
    x = x.substr(1);
  }
  return x;
}

int getTypeSize(SpVcVMTypeDescriptor *tp) {
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_INT32)
    return 4;
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_FLOAT32)
    return 4;
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_UNSIGNED32)
    return 4;
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_BOOL)
    return 1;
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_VECTOR) {
    auto sz = tp->size;
    auto chSz = getTypeSize(tp->children[0]);
    return sz * chSz;
  }
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_STRUCT) {
    auto sz = 0;
    for (auto &ch : tp->children) {
      sz += getTypeSize(ch);
    }
    return sz;
  }
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_MATRIX) {
    auto sz = getTypeSize(tp->children[0]);
    return sz * tp->size;
  }
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_ARRAY) {
    auto sz = getTypeSize(tp->children[0]);
    return sz * tp->size;
  }
  ifritError("Unknown type size");
}

} // namespace IrUtil

namespace DefinitionPass {
// Structural Control Flow
DEF_PASS(OpSelectionMerge) {
  auto startPc = irg->getActiveBlock()->startingPc;
  ctx->cfgInd = SPVC_STRUCTCFG_SELECTION_MERGE;
  auto tgt = GET_PARAM(0);
  tgt->blockBelong->blockType.push_back({SPVC_BLOCK_SELECTION_MERGE, startPc});
  auto act = irg->getActiveBlock();
  act->blockType.push_back({SPVC_BLOCK_SELECTION_HEADER, startPc});
}
DEF_PASS(OpLoopMerge) {
  auto startPc = irg->getActiveBlock()->startingPc;
  ctx->cfgInd = SPVC_STRUCTCFG_LOOP_MERGE;
  auto mergeTgt = GET_PARAM(0);
  auto contTgt = GET_PARAM(1);
  mergeTgt->blockBelong->blockType.push_back({SPVC_BLOCK_LOOP_MERGE, startPc});
  contTgt->blockBelong->blockType.push_back(
      {SPVC_BLOCK_LOOP_CONTINUE, startPc});
  auto act = irg->getActiveBlock();
  act->blockType.push_back({SPVC_BLOCK_LOOP_HEADER, startPc});
}
DEF_PASS(OpBranchConditional) {
  auto startPc = irg->getActiveBlock()->startingPc;
  auto trueJmp = GET_PARAM(1);
  auto falseJmp = GET_PARAM(2);
  if (ctx->cfgInd == SPVC_STRUCTCFG_SELECTION_MERGE) {
    trueJmp->blockBelong->blockType.push_back(
        {SPVC_BLOCK_SELECTION_BODY_FIRST, startPc});
    falseJmp->blockBelong->blockType.push_back(
        {SPVC_BLOCK_SELECTION_BODY_SECOND, startPc});
  } else if (ctx->cfgInd == SPVC_STRUCTCFG_LOOP_MERGE) {
    auto act = irg->getActiveBlock();
    act->blockType.push_back({SPVC_BLOCK_LOOP_BREAK, startPc});

    SpVcBlockTypeRecord rec = {SPVC_BLOCK_LOOP_MERGE, startPc};
    auto trueFind = std::find(trueJmp->blockBelong->blockType.begin(),
                              trueJmp->blockBelong->blockType.end(), rec);
    if (trueFind == trueJmp->blockBelong->blockType.end())
      trueJmp->blockBelong->blockType.push_back(
          {SPVC_BLOCK_LOOP_BODY, startPc});
    auto falseFind = std::find(falseJmp->blockBelong->blockType.begin(),
                               falseJmp->blockBelong->blockType.end(), rec);
    if (falseFind == falseJmp->blockBelong->blockType.end())
      falseJmp->blockBelong->blockType.push_back(
          {SPVC_BLOCK_LOOP_BODY, startPc});
  }
  irg->getActiveBlock()->cfgSuccessor.push_back(trueJmp->blockBelong);
  trueJmp->blockBelong->cfgPredecessor.push_back(irg->getActiveBlock());
  falseJmp->blockBelong->cfgPredecessor.push_back(irg->getActiveBlock());
  ctx->cfgInd = SPVC_STRUCTCFG_NONE;
}
DEF_PASS(OpBranch) {
  auto startPc = irg->getActiveBlock()->startingPc;
  auto jmp = GET_PARAM(0);
  if (ctx->cfgInd == SPVC_STRUCTCFG_SELECTION_MERGE) {
    EMIT_ERROR("Selection merge cannot precedes OpBranch");
  } else if (ctx->cfgInd == SPVC_STRUCTCFG_LOOP_MERGE) {
    auto act = irg->getActiveBlock();
    act->blockType.push_back({SPVC_BLOCK_LOOP_BREAK, startPc});

    SpVcBlockTypeRecord rec = {SPVC_BLOCK_LOOP_MERGE, startPc};
    auto jmpFind = std::find(jmp->blockBelong->blockType.begin(),
                             jmp->blockBelong->blockType.end(), rec);
    if (jmpFind == jmp->blockBelong->blockType.end())
      jmp->blockBelong->blockType.push_back({SPVC_BLOCK_LOOP_BODY, startPc});
  } else {
    // Check if LOOP BREAK
    for (int i = 0; i < jmp->blockBelong->blockType.size(); i++) {
      if (jmp->blockBelong->blockType[i].blockType == SPVC_BLOCK_LOOP_MERGE) {
        auto act = irg->getActiveBlock();
        act->blockType.push_back({SPVC_BLOCK_LOOP_BREAK,
                                  jmp->blockBelong->blockType[i].progCounter});
        break;
      }
    }
  }
  irg->getActiveBlock()->cfgSuccessor.push_back(jmp->blockBelong);
  jmp->blockBelong->cfgPredecessor.push_back(irg->getActiveBlock());
  ctx->cfgInd = SPVC_STRUCTCFG_NONE;
}
DEF_PASS(OpSwitch) {
  auto startPc = irg->getActiveBlock()->startingPc;
  for (int i = 1; i < params.size(); i++) {
    auto target = GET_PARAM(i);
    target->blockBelong->blockType.push_back(
        {SPVC_BLOCK_SELECTION_BODY_SWITCH, startPc});
    irg->getActiveBlock()->cfgSuccessor.push_back(target->blockBelong);
    target->blockBelong->cfgPredecessor.push_back(irg->getActiveBlock());
  }
  ctx->cfgInd = SPVC_STRUCTCFG_NONE;
}
DEF_PASS(OpLabel) {
  auto thisBlock = GET_PARAM(0)->blockBelong;
  irg->pushActiveBlock(thisBlock);
}

// Type Definitions
DEF_PASS(OpTypeFloat) {
  auto tgt = GET_PARAM(0);
  auto bits = GET_PARAM_SCALAR(1);
  tgt->flag |= SPVC_VARIABLE_TYPE;
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  if (bits == 64)
    tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_FLOAT64;
  if (bits == 32)
    tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_FLOAT32;
  if (bits == 16)
    tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_FLOAT16;
  if (bits == 8)
    tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_FLOAT8;
}
DEF_PASS(OpTypeInt) {
  auto tgt = GET_PARAM(0);
  auto bits = GET_PARAM_SCALAR(1);
  auto sign = GET_PARAM_SCALAR(2);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->flag |= SPVC_VARIABLE_TYPE;
  if (sign == 0) {
    if (bits == 64)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_UNSIGNED64;
    if (bits == 32)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_UNSIGNED32;
    if (bits == 16)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_UNSIGNED16;
    if (bits == 8)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_UNSIGNED8;
  } else {
    if (bits == 64)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_INT64;
    if (bits == 32)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_INT32;
    if (bits == 16)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_INT16;
    if (bits == 8)
      tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_INT8;
  }
}
DEF_PASS(OpTypePointer) {
  auto tgt = GET_PARAM(0);
  auto storage = GET_PARAM_SCALAR(1);
  auto tp = GET_PARAM(2);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_POINTER;
  tgt->tp->children.push_back(tp->tp.get());
  tgt->tp->storageClass = storage;
  tgt->flag |= SPVC_VARIABLE_TYPE;
  tgt->tpRef = tp;
}
DEF_PASS(OpTypeVector) {
  auto tgt = GET_PARAM(0);
  auto tp = GET_PARAM(1);
  auto size = GET_PARAM_SCALAR(2);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_VECTOR;
  tgt->tp->children.push_back(tp->tp.get());
  tgt->tp->size = size;
  tgt->flag |= SPVC_VARIABLE_TYPE;
}
DEF_PASS(OpTypeBool) {
  auto tgt = GET_PARAM(0);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_BOOL;
  tgt->flag |= SPVC_VARIABLE_TYPE;
}
DEF_PASS(OpTypeArray) {
  auto tgt = GET_PARAM(0);
  auto tp = GET_PARAM(1);
  auto size = GET_PARAM(2);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_ARRAY;
  tgt->tp->children.push_back(tp->tp.get());
  tgt->tp->size = size->constant->value[0];
  tgt->flag |= SPVC_VARIABLE_TYPE;
}
DEF_PASS(OpTypeVoid) {
  auto tgt = GET_PARAM(0);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_VOID;
  tgt->flag |= SPVC_VARIABLE_TYPE;
}
DEF_PASS(OpTypeStruct) {
  auto tgt = GET_PARAM(0);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_STRUCT;
  for (int i = 1; i < params.size(); i++) {
    auto tp = GET_PARAM(i);
    tgt->tp->children.push_back(tp->tp.get());
  }
  tgt->flag |= SPVC_VARIABLE_TYPE;
}
DEF_PASS(OpTypeFunction) {
  auto tgt = GET_PARAM(1);
  auto retTp = GET_PARAM(0);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_FUNCTION;
  tgt->tp->children.push_back(retTp->tp.get());
  for (int i = 2; i < params.size(); i++) {
    auto tp = GET_PARAM(i);
    tgt->tp->children.push_back(tp->tp.get());
  }
  tgt->flag |= SPVC_VARIABLE_TYPE;
}
DEF_PASS(OpReturn) {
  auto startPc = irg->getActiveBlock()->startingPc;
  irg->getActiveBlock()->blockType.push_back({SPVC_BLOCK_RETURN, startPc});
}
DEF_PASS(OpTypeImage) {
  auto tgt = GET_PARAM(0);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_IMAGE;
  tgt->flag |= SPVC_VARIABLE_TYPE;
  EMIT_WARN("Not implemented: OpTypeImage");
}
DEF_PASS(OpTypeSampler) {
  auto tgt = GET_PARAM(0);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_SAMPLER;
  tgt->flag |= SPVC_VARIABLE_TYPE;
  EMIT_WARN("Not implemented: OpTypeSampler");
}
DEF_PASS(OpTypeSampledImage) {
  auto tgt = GET_PARAM(0);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_SAMPLED_IMAGE;
  tgt->flag |= SPVC_VARIABLE_TYPE;
  EMIT_WARN("Not implemented: OpTypeSampledImage");
}
DEF_PASS(OpTypeMatrix) {
  auto tgt = GET_PARAM(0);
  auto tp = GET_PARAM(1);
  auto size = GET_PARAM_SCALAR(2);
  tgt->tp = std::make_unique<SpVcVMTypeDescriptor>();
  tgt->tp->type = SpVcVMTypeEnum::SPVC_TYPE_MATRIX;
  tgt->tp->children.push_back(tp->tp.get());
  tgt->tp->size = size;
  tgt->flag |= SPVC_VARIABLE_TYPE;
}

// Constant Definition
DEF_PASS(OpConstant) {
  auto tgt = GET_PARAM(1);
  auto tp = GET_PARAM(0);
  auto val = GET_PARAM_SCALAR(2);
  tgt->tpRef = tp;
  tgt->constant = std::make_unique<SpVcVMGenConstant>();
  tgt->constant->tpRef = tp->tp.get();
  tgt->constant->value.push_back(val);
  tgt->flag |= SPVC_VARIABLE_CONSTANT;
}
DEF_PASS(OpConstantComposite) {
  auto tgt = GET_PARAM(1);
  auto tp = GET_PARAM(0);
  tgt->constant = std::make_unique<SpVcVMGenConstant>();
  tgt->tpRef = tp;
  for (int i = 2; i < params.size(); i++) {
    auto val = GET_PARAM(i);
    if (val->constant.get() == nullptr) {
      EMIT_ERROR("Invalid constant:", GET_PARAM_SCALAR(i));
    }
    tgt->constant->children.push_back(val->constant.get());
  }
  tgt->flag |= SPVC_VARIABLE_CONSTANT;
}

// Mode Setting
DEF_PASS(OpCapability) { ctx->capabilities.push_back(GET_PARAM_SCALAR(0)); }
DEF_PASS(OpExtInstImport) {
  auto tgt = GET_PARAM(0);
  int strParam = 1;
  tgt->flag |= SPVC_VARIABLE_EXTINST_IMPORT;
  auto str = GET_PARAM_STR(strParam);
  tgt->name = str;
}
DEF_PASS(OpMemoryModel) {
  ctx->memoryModel.memoryModel = GET_PARAM_SCALAR(0);
  ctx->memoryModel.addressingModel = GET_PARAM_SCALAR(1);
}
DEF_PASS(OpEntryPoint) {
  auto entryType = GET_PARAM_SCALAR(0);
  auto pt = GET_PARAM_SCALAR(1);
  int literalx = 2;
  auto name = GET_PARAM_STR(literalx);
  SpVcVMEntryPoint ept;
  ept.execModel = entryType;
  ept.entryPoint = pt;
  ept.name = name;
  for (int i = literalx; i < params.size(); i++) {
    auto val = GET_PARAM_SCALAR(i);
    ept.input.push_back(val);
  }
  ctx->entryPoints.push_back(ept);
}
DEF_PASS(OpExecutionMode) {
  // Ignore
}

DEF_PASS(OpFunction) {
  auto target = GET_PARAM(1);
  if (target->funcBelong == nullptr) {
    EMIT_ERROR("Invalid function");
  }
  ctx->activeFuncEnv = target->funcBelong;
}
DEF_PASS(OpFunctionEnd) { ctx->activeFuncEnv = nullptr; }

// Debug Settings
DEF_PASS(OpSource) {
  // Ignore
}
DEF_PASS(OpSourceExtension) {
  // Ignore
}
DEF_PASS(OpName) {
  auto tgt = GET_PARAM(0);
  int strParam = 1;
  auto str = GET_PARAM_STR(strParam);
  tgt->name = str;
}
DEF_PASS(OpMemberName) { EMIT_WARN("Not implemented: OpMemberName"); }

// Variable Definition
DEF_PASS(OpVariable) {
  auto tgt = GET_PARAM(1);
  auto tp = GET_PARAM(0);
  tgt->tpRef = tp->tpRef;
  tgt->flag |= SPVC_VARIABLE_VAR;
  tgt->storageClass = GET_PARAM_SCALAR(2);
  if (tgt->storageClass == spv::StorageClass::StorageClassUniform ||
      tgt->storageClass == spv::StorageClass::StorageClassUniformConstant) {
    tgt->isAllQuad = true;
  }
  tgt->blockBelong = irg->getActiveBlock();
}

// Annotations
DEF_PASS(OpDecorate) {
  auto target = GET_PARAM(0);
  auto dec = GET_PARAM_SCALAR(1);
  if (target->descSet == nullptr) {
    target->descSet = std::make_unique<SpVcVMDecoration>();
  }
  if (dec == spv::Decoration::DecorationDescriptorSet) {
    target->descSet->descriptorSet = GET_PARAM_SCALAR(2);
  } else if (dec == spv::Decoration::DecorationLocation) {
    target->descSet->location = GET_PARAM_SCALAR(2);
  } else if (dec == spv::Decoration::DecorationBinding) {
    target->descSet->binding = GET_PARAM_SCALAR(2);
  } else {
    EMIT_WARN("Unknown decoration: ", dec);
  }
}
DEF_PASS(OpMemberDecorate) { EMIT_WARN("Not implemented: OpMemberDecorate"); }

DEF_PASS(OpAccessChain) {
  auto tgt = GET_PARAM(1);
  auto tp = GET_PARAM(0);
  tgt->tpRef = tp->tpRef;
  tgt->blockBelong = irg->getActiveBlock();
  tgt->flag |= SPVC_VARIABLE_TEMP;
  tgt->flag |= SPVC_VARIABLE_ACCESS_CHAIN;
}

// Universal Definition
DEF_PASS(DefOpUniversal) {
  auto tgt = GET_PARAM(1);
  auto tp = GET_PARAM(0);
  tgt->tpRef = tp;
  tgt->blockBelong = irg->getActiveBlock();
  tgt->flag |= SPVC_VARIABLE_TEMP;
}
DEF_PASS(DefOpIgnore) {}
} // namespace DefinitionPass

namespace ConversionPass {
LLVM::SpVcLLVMArgument *makeMaskVar(SpVcQuadGroupedIRGenerator *irg) {
  auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
      irg->allocateLlvmVarName()));
  auto tp = irg->addIr(std::make_unique<LLVM::SpVcLLVMTypeInt32>());
  auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
  return q;
}

LLVM::SpVcLLVMArgument *makeBoolVar(SpVcQuadGroupedIRGenerator *irg) {
  auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
      irg->allocateLlvmVarName()));
  auto tp = irg->addIr(std::make_unique<LLVM::SpVcLLVMTypeBool>());
  auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
  return q;
}

LLVM::SpVcLLVMArgument *makeOneImm(SpVcQuadGroupedIRGenerator *irg) {
  auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
  auto tp = irg->addIr(std::make_unique<LLVM::SpVcLLVMTypeInt32>());
  auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
  return q;
}

LLVM::SpVcLLVMType *spirvGetImmediateType(int id, int invo,
                                          SpVcQuadGroupedIRGenerator *irg) {
  return irg->getVariableSafe(id)->tpRef->tp->llvmType;
}
LLVM::SpVcLLVMArgument *spirvImmediateLoad(int id, int invo,
                                           SpVcQuadGroupedIRGenerator *irg) {
  auto var = irg->getVariableSafe(id);
  if (var->flag & (SPVC_VARIABLE_VAR | SPVC_VARIABLE_TEMP)) {
    if (var->isAllQuad)
      invo = 0;
    auto cVar = var->llvmVarName[invo].arg;
    auto cVarReg = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto cVarType = spirvGetImmediateType(id, invo, irg);
    auto cVarTypeArg =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(cVarType, cVarReg));
    auto cVarLoad = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(cVarTypeArg, cVar),
        irg->getActiveBlock());
    return cVarTypeArg;
  } else if (var->flag & SPVC_VARIABLE_CONSTANT) {
    auto cVar = var->constant->arg;
    return cVar;
  } else {
    EMIT_ERROR("Invalid immediate load");
    return nullptr;
  }
}
LLVM::SpVcLLVMArgument *
spirvCreateVarWithSameType(int id, int invo, SpVcQuadGroupedIRGenerator *irg) {
  auto var = irg->getVariableSafe(id);
  auto cVar = var->llvmVarName[invo].arg;
  auto cVarReg = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
      irg->allocateLlvmVarName()));
  auto cVarType = spirvGetImmediateType(id, invo, irg);
  auto cVarTypeArg =
      irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(cVarType, cVarReg));
  return cVarTypeArg;
}

LLVM::SpVcLLVMArgument *
spirvCreateVarWithType(SpVcVMTypeDescriptor *desc,
                       SpVcQuadGroupedIRGenerator *irg) {
  auto cVarReg = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
      irg->allocateLlvmVarName()));
  auto cVarType = desc->llvmType;
  auto cVarTypeArg =
      irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(cVarType, cVarReg));
  return cVarTypeArg;
}

void spirvImmediateMaskedStore(int id, int invo, LLVM::SpVcLLVMArgument *val,
                               SpVcQuadGroupedIRGenerator *irg) {
  auto var = irg->getVariableSafe(id);
  auto cVar = var->llvmVarName[invo].arg;

  auto actMask = irg->getActiveBlock()
                     ->stackBelong->masks.activeExecMask.back()
                     ->llvmVarName[invo]
                     .arg;
  auto actMaskReg = makeMaskVar(irg);
  // Load actMask -> actMaskReg
  auto actMaskLoad = irg->addIrB(
      std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(actMaskReg, actMask),
      irg->getActiveBlock());

  auto imm1 = makeOneImm(irg);
  auto cmpReg = makeBoolVar(irg);
  auto cmp = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                             cmpReg, actMaskReg, imm1, "icmp eq"),
                         irg->getActiveBlock());

  auto cVarArg = spirvImmediateLoad(id, invo, irg);
  auto selRes = spirvCreateVarWithSameType(id, invo, irg);
  auto sel = irg->addIrB(
      std::make_unique<LLVM::SpVcLLVMIns_Select>(selRes, cmpReg, val, cVarArg),
      irg->getActiveBlock());
  auto cVarStore = irg->addIrB(
      std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(cVar, selRes),
      irg->getActiveBlock());
}

void foreachInvo(std::function<void(int)> func,
                 SpVcQuadGroupedIRGenerator *irg) {
  for (int i = 0; i < irg->getQuads(); i++) {
    func(i);
  }
}

CONV_PASS(DefOpIgnore) {
  // Pass
}

CONV_PASS(OpLabel) {
  auto CONTAIN_TAG = [&](int reqType) {
    auto p = GET_PARAM(0);
    for (auto &s : p->blockBelong->blockType) {
      if (s.blockType == reqType)
        return true;
    }
    return false;
  };
  using MASK_TYPE = LLVM::SpVcLLVMTypeInt32;
  auto MAKE_MASK = [&]() {
    auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
    return q;
  };

  auto MAKE_LABEL = [&]() {
    auto q = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMLabelName>(irg->allocateLlvmVarName()),
        irg->getActiveBlock());
    return q;
  };

  auto MAKE_MASK_ALLOC = [&](LLVM::SpVcLLVMArgument *q) {
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto p = irg->addIrF(std::make_unique<LLVM::SpVcLLVMIns_Alloca>(q, tp),
                         irg->getActiveBlock());
  };

  auto MAKE_MASK_INIT = [&](LLVM::SpVcLLVMArgument *q) {
    auto immOne =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto sq = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, immOne));
    auto p =
        irg->addIrF(std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(q, sq),
                    irg->getActiveBlock());
  };

  auto MAKE_MASK_INIT_LOCAL = [&](LLVM::SpVcLLVMArgument *q) {
    auto immOne =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto sq = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, immOne));
    auto p =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(q, sq),
                    irg->getActiveBlock());
  };

  auto MAKE_MASK_AND = [&](LLVM::SpVcLLVMArgument *dest,
                           LLVM::SpVcLLVMArgument *a,
                           LLVM::SpVcLLVMArgument *b) {
    auto rDest = MAKE_MASK();
    auto rA = MAKE_MASK();
    auto rB = MAKE_MASK();
    // auto p =
    // irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rDest,
    // dest), irg->getActiveBlock());
    auto q =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rA, a),
                    irg->getActiveBlock());
    auto s =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rB, b),
                    irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(rDest, rA, rB, "and"),
        irg->getActiveBlock());
    auto u = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(dest, rDest),
        irg->getActiveBlock());
  };

  auto MAKE_MASK_NEGATE = [&](LLVM::SpVcLLVMArgument *dest,
                              LLVM::SpVcLLVMArgument *a) {
    auto rDest = MAKE_MASK();
    auto rA = MAKE_MASK();
    auto allOneImm =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
    auto allOneImmType = irg->addIr(std::make_unique<MASK_TYPE>());
    auto allOneArg = irg->addIr(
        std::make_unique<LLVM::SpVcLLVMArgument>(allOneImmType, allOneImm));

    // auto p =
    // irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rDest,
    // dest), irg->getActiveBlock());
    auto q =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rA, a),
                    irg->getActiveBlock());
    auto minus = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                                 rDest, allOneArg, rA, "sub"),
                             irg->getActiveBlock());
    auto u = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(dest, rDest),
        irg->getActiveBlock());
  };

  auto FORALL_CHAN = [&](std::function<void(int)> f) {
    for (int i = 0; i < irg->getQuads(); i++)
      f(i);
  };

  auto FIND_STRUCT_PRED = [&](SpVcVMGenBlock *b) {
    int found = 0;
    SpVcVMGenBlock *pred = nullptr;
    for (int i = 0; i < b->cfgPredecessor.size(); i++) {
      SpVcBlockTypeRecord rp = {SPVC_BLOCK_LOOP_CONTINUE, b->startingPc};
      if (std::find(b->cfgPredecessor[i]->blockType.begin(),
                    b->cfgPredecessor[i]->blockType.end(),
                    rp) != b->cfgPredecessor[i]->blockType.end()) {
        continue;
      }
      if (found) {
        EMIT_ERROR("Multiple struct preds", found);
      }
      found = 1;
      pred = b->cfgPredecessor[i];
    }
    return pred;
  };

#define MASK_BR(k, v, x) v.branchMask[k]->llvmVarName[(x)].arg
#define MASK_BR_LAST(v, x) v.branchMask.back()->llvmVarName[(x)].arg
#define MASK_EXEC(v, x) v.execMask->llvmVarName[(x)].arg
#define MASK_RET(v, x) v.returnMask->llvmVarName[(x)].arg
#define MASK_ACT(k, v, x) v.activeExecMask[k]->llvmVarName[(x)].arg
#define MASK_ACT_LAST(v, x) v.activeExecMask.back()->llvmVarName[(x)].arg
#define MASK_BREAK(v, x) v.breakMask->llvmVarName[(x)].arg
#define MASK_CONT(v, x) v.continueMask->llvmVarName[(x)].arg

  auto MASK_ACT_LAST_PREV = [&](SpVcVMExecutionMask &v, int x) {
    if (v.activeExecMask.size() < 2) {
      EMIT_ERROR("Invalid active mask");
    }
    return v.activeExecMask[v.activeExecMask.size() - 2]->llvmVarName[x].arg;
  };

  auto curBx = GET_PARAM(0)->blockBelong;
  irg->pushActiveBlock(curBx);
  bool stackSpecified = false;

  // Handle Top Block
  if constexpr (true) {
    auto curB = GET_PARAM(0)->blockBelong;
    if (!FIND_STRUCT_PRED(curB)) {
      // Jump to block label
      auto p = curB->llvmLabel;
      auto q = irg->addIrBPre(std::make_unique<LLVM::SpVcLLVMIns_Br>(p), curB);
    }
  }

  // Loop Merge : Stack pop
  if (CONTAIN_TAG(SPVC_BLOCK_LOOP_MERGE)) {
    irg->popNewStack();
    auto sp = irg->getActiveStack();
    auto curB = GET_PARAM(0)->blockBelong;
    curB->stackBelong = sp;
    stackSpecified = true;
  }

  // Selection Merge: Br Stack Pop
  if (CONTAIN_TAG(SPVC_BLOCK_SELECTION_MERGE)) {
    auto sp = irg->getActiveStack();
    auto curB = GET_PARAM(0)->blockBelong;
    curB->stackBelong = sp;
    sp->ifStackSize--;
    stackSpecified = true;
  }

  // Loop header : Loop stack push
  if (CONTAIN_TAG(SPVC_BLOCK_LOOP_HEADER)) {
    irg->pushNewStack();
    irg->pushActiveBlock(GET_PARAM(0)->blockBelong);
    auto sp = irg->getActiveStack();
    auto curB = GET_PARAM(0)->blockBelong;
    curB->stackBelong = sp;
    stackSpecified = true;
    sp->ifStackSize = 1;

    sp->masks.execMask = irg->createExecutionMaskVar();
    sp->masks.breakMask = irg->createExecutionMaskVar();
    sp->masks.continueMask = irg->createExecutionMaskVar();
    sp->masks.activeExecMask.push_back(irg->createExecutionMaskVar());
    sp->masks.branchMask.push_back(irg->createExecutionMaskVar());

    FORALL_CHAN([&](int x) {
      MASK_EXEC(sp->masks, x) = MAKE_MASK();
      MAKE_MASK_ALLOC(MASK_EXEC(sp->masks, x));
      MASK_BREAK(sp->masks, x) = MAKE_MASK();
      MAKE_MASK_ALLOC(MASK_BREAK(sp->masks, x));
      MASK_CONT(sp->masks, x) = MAKE_MASK();
      MAKE_MASK_ALLOC(MASK_CONT(sp->masks, x));
      MASK_ACT_LAST(sp->masks, x) = MAKE_MASK();
      MAKE_MASK_ALLOC(MASK_ACT_LAST(sp->masks, x));
      MASK_BR_LAST(sp->masks, x) = MAKE_MASK();
      MAKE_MASK_ALLOC(MASK_BR_LAST(sp->masks, x));

      MAKE_MASK_INIT_LOCAL(MASK_EXEC(sp->masks, x));
      MAKE_MASK_INIT_LOCAL(MASK_BREAK(sp->masks, x));
      MAKE_MASK_INIT_LOCAL(MASK_CONT(sp->masks, x));
      MAKE_MASK_INIT_LOCAL(MASK_ACT_LAST(sp->masks, x));
      MAKE_MASK_INIT_LOCAL(MASK_BR_LAST(sp->masks, x));
    });

    auto predecessor = FIND_STRUCT_PRED(curB);
    if (predecessor) {
      auto predSp = predecessor->stackBelong;
      FORALL_CHAN([&](int x) {
        auto predMask = MASK_ACT_LAST(predSp->masks, x);
        auto curMask = MASK_EXEC(sp->masks, x);
        auto curAct = MASK_ACT_LAST(sp->masks, x);
        MAKE_MASK_AND(curAct, predMask, curMask);
        MAKE_MASK_AND(curMask, predMask, curMask);
      });
    }
    FORALL_CHAN([&](int x) {
      auto curMask = MASK_EXEC(sp->masks, x);
      auto retMask = curB->funcBelong->returnMask->llvmVarName[x].arg;
      auto curAct = MASK_ACT_LAST(sp->masks, x);
      MAKE_MASK_AND(curMask, retMask, curMask);
      MAKE_MASK_AND(curMask, retMask, curMask);
    });
    // Create Label
    curB->contLabel = MAKE_LABEL();
  }

  // Loop Continue : Cont mask recover
  if (CONTAIN_TAG(SPVC_BLOCK_LOOP_CONTINUE)) {
    auto sp = irg->getActiveStack();
    auto curB = GET_PARAM(0)->blockBelong;
    curB->stackBelong = sp;
    stackSpecified = true;
    FORALL_CHAN([&](int x) {
      MAKE_MASK_INIT_LOCAL(MASK_CONT(sp->masks, x));
      auto initMask = MASK_EXEC(sp->masks, x);
      auto curBreak = MASK_BREAK(sp->masks, x);
      auto curAct = MASK_ACT_LAST(sp->masks, x);
      auto retMask = curB->funcBelong->returnMask->llvmVarName[x].arg;

      MAKE_MASK_AND(curAct, initMask, curBreak);
      MAKE_MASK_AND(curAct, retMask, curAct);
    });
  }

  // Selection Header : Br Stack Push
  if (CONTAIN_TAG(SPVC_BLOCK_SELECTION_HEADER)) {
    auto sp = irg->getActiveStack();
    auto curB = GET_PARAM(0)->blockBelong;
    stackSpecified = true;
    sp->ifStackSize++;
    curB->stackBelong = sp;
    sp->masks.branchMask.push_back(irg->createExecutionMaskVar());
    sp->masks.activeExecMask.push_back(irg->createExecutionMaskVar());
    FORALL_CHAN([&](int x) {
      MASK_BR_LAST(sp->masks, x) = MAKE_MASK();
      MAKE_MASK_ALLOC(MASK_BR_LAST(sp->masks, x));
      MAKE_MASK_INIT(MASK_BR_LAST(sp->masks, x));
      MASK_ACT_LAST(sp->masks, x) = MAKE_MASK();
      MAKE_MASK_ALLOC(MASK_ACT_LAST(sp->masks, x));
      MAKE_MASK_INIT(MASK_ACT_LAST(sp->masks, x));

      MAKE_MASK_AND(MASK_ACT_LAST(sp->masks, x), MASK_ACT_LAST(sp->masks, x),
                    MASK_ACT_LAST_PREV(sp->masks, x));

      auto retMask = curB->funcBelong->returnMask->llvmVarName[x].arg;
      auto curAct = MASK_ACT_LAST(sp->masks, x);
      MAKE_MASK_AND(curAct, retMask, curAct);
    });
  }

  // Selection Body: Reverse Br Mask & And with LastActPrev
  if (CONTAIN_TAG(SPVC_BLOCK_SELECTION_BODY_SECOND)) {
    auto sp = irg->getActiveStack();
    auto curB = GET_PARAM(0)->blockBelong;
    curB->stackBelong = sp;
    auto pred = curB->cfgPredecessor[0];
    stackSpecified = true;
    auto predSp = pred->stackBelong;
    FORALL_CHAN([&](int x) {
      auto predMask = MASK_ACT_LAST_PREV(predSp->masks, x);
      auto curMask = MASK_BR_LAST(sp->masks, x);
      auto curAct = MASK_ACT_LAST(sp->masks, x);
      MAKE_MASK_NEGATE(curMask, curMask);
      MAKE_MASK_AND(curAct, predMask, curMask);

      auto retMask = curB->funcBelong->returnMask->llvmVarName[x].arg;
      MAKE_MASK_AND(curAct, retMask, curAct);
    });
  }

  // Otherwise, specify stack info
  if (stackSpecified == false) {
    auto sp = irg->getActiveStack();
    auto curB = GET_PARAM(0)->blockBelong;
    curB->stackBelong = sp;

    FORALL_CHAN([&](int x) {
      auto curAct = MASK_ACT_LAST(sp->masks, x);
      auto retMask = curB->funcBelong->returnMask->llvmVarName[x].arg;
      MAKE_MASK_AND(curAct, retMask, curAct);
    });
  }
}
CONV_PASS(OpSwitch) { EMIT_ERROR("not supported opswitch"); }
CONV_PASS(OpBranch) {
  auto curB = irg->getActiveBlock();
  auto CONTAIN_TAG = [&](int reqType) {
    auto p = irg->getActiveBlock();
    for (auto &s : p->blockType) {
      if (s.blockType == reqType)
        return true;
    }
    return false;
  };
  auto FORALL_CHAN = [&](std::function<void(int)> f) {
    for (int i = 0; i < irg->getQuads(); i++)
      f(i);
  };

  using MASK_TYPE = LLVM::SpVcLLVMTypeInt32;
  auto MAKE_MASK = [&]() {
    auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
    return q;
  };
  auto MAKE_MASK_AND = [&](LLVM::SpVcLLVMArgument *dest,
                           LLVM::SpVcLLVMArgument *a,
                           LLVM::SpVcLLVMArgument *b) {
    auto rDest = MAKE_MASK();
    auto rA = MAKE_MASK();
    auto rB = MAKE_MASK();
    // auto p = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_Load>(rDest,
    // dest), irg->getActiveBlock());
    auto q =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rA, a),
                    irg->getActiveBlock());
    auto s =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rB, b),
                    irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(rDest, rA, rB, "and"),
        irg->getActiveBlock());
    auto u = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(dest, rDest),
        irg->getActiveBlock());
  };
  auto MAKE_MASK_AND_TO_REG = [&](LLVM::SpVcLLVMArgument *dest,
                                  LLVM::SpVcLLVMArgument *a,
                                  LLVM::SpVcLLVMArgument *b) {
    auto rA = MAKE_MASK();
    auto rB = MAKE_MASK();
    auto q =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rA, a),
                    irg->getActiveBlock());
    auto s =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rB, b),
                    irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(dest, rA, rB, "and"),
        irg->getActiveBlock());
  };
  auto MAKE_MASK_OR_REG = [&](LLVM::SpVcLLVMArgument *dest,
                              LLVM::SpVcLLVMArgument *a,
                              LLVM::SpVcLLVMArgument *b) {
    auto rB = MAKE_MASK();
    auto s =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rB, b),
                    irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(dest, a, rB, "or"),
        irg->getActiveBlock());
  };
  auto target = GET_PARAM(0);

  if (CONTAIN_TAG(SPVC_BLOCK_SELECTION_BODY_FIRST)) {
    // This branch should be ignored
  }
  if (CONTAIN_TAG(SPVC_BLOCK_SELECTION_BODY_SECOND)) {
    // This branch should be ignored too
  }
  if (CONTAIN_TAG(SPVC_BLOCK_LOOP_CONTINUE)) {
    // Jump back to loop header
    auto sp = curB->stackBelong;
    auto curB = GET_PARAM(0)->blockBelong;
    auto label = curB->contLabel;
    // Create BR
    auto br = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_Br>(label),
                          irg->getActiveBlock());
  }
  if (CONTAIN_TAG(SPVC_BLOCK_LOOP_BREAK)) {
    // Fetch BR_MASK, For 1 in BR_MASK, set break mask to 0
    auto sp = curB->stackBelong;
    auto curB = irg->getActiveBlock();
    FORALL_CHAN([&](int x) {
      auto brMask = MASK_BR_LAST(sp->masks, x);
      auto brMaskReg = MAKE_MASK();
      auto brMaskRegNeg = MAKE_MASK();
      auto breakMask = MASK_BREAK(sp->masks, x);
      auto breakMaskReg = MAKE_MASK();
      auto p = irg->addIrB(
          std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(brMaskReg, brMask),
          irg->getActiveBlock());
      auto p1 = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(
                                breakMaskReg, breakMask),
                            irg->getActiveBlock());

      auto immOne =
          irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
      auto tp = irg->addIr(std::make_unique<LLVM::SpVcLLVMTypeInt32>());
      auto sq =
          irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, immOne));
      auto ps = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                                brMaskRegNeg, sq, brMaskReg, "isub"),
                            irg->getActiveBlock());
      auto ps1 =
          irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                          breakMaskReg, breakMaskReg, brMaskRegNeg, "and"),
                      irg->getActiveBlock());

      // Store
      auto p2 = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(
                                breakMask, breakMaskReg),
                            irg->getActiveBlock());

      // AND ACT_MASK
      auto actMask = MASK_ACT_LAST(sp->masks, x);
      MAKE_MASK_AND(actMask, actMask, breakMask);
    });

    // IF all zero, jump to merge block
    auto mergeBlock = GET_PARAM(0)->blockBelong;
    auto mergeLabel = mergeBlock->llvmLabel;
    LLVM::SpVcLLVMArgument *maskOrToken[4] = {MAKE_MASK(), MAKE_MASK(),
                                              MAKE_MASK(), MAKE_MASK()};
    FORALL_CHAN([&](int x) {
      if (x == 0) {
        MAKE_MASK_AND_TO_REG(maskOrToken[x], MASK_ACT_LAST(sp->masks, x),
                             MASK_ACT_LAST(sp->masks, x));
      } else {
        MAKE_MASK_OR_REG(maskOrToken[x], maskOrToken[x - 1],
                         MASK_ACT_LAST(sp->masks, x));
      }
    });
    auto tpZero = irg->addIr(std::make_unique<MASK_TYPE>());
    auto immZero =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(0));
    auto immZeroArg =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpZero, immZero));
    auto cmpRetTp = irg->addIr(std::make_unique<LLVM::SpVcLLVMTypeBool>());
    auto cmpRetName =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
            irg->allocateLlvmVarName()));
    auto cmpRet = irg->addIr(
        std::make_unique<LLVM::SpVcLLVMArgument>(cmpRetTp, cmpRetName));
    auto cmp = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                               cmpRet, maskOrToken[3], immZeroArg, "icmp eq"),
                           irg->getActiveBlock());

    auto skipLabel = irg->addIr(
        std::make_unique<LLVM::SpVcLLVMLabelName>(irg->allocateLlvmVarName()));
    auto br = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_BrCond>(
                              cmpRet, mergeLabel, skipLabel),
                          irg->getActiveBlock());
    irg->addIrBExist(skipLabel, irg->getActiveBlock());
  }
}
CONV_PASS(OpBranchConditional) {
  // Stores conds into MASK_BR_LAST
  // If this is a loop header, BR jumps to body or merge block

  auto curB = GET_PARAM(0)->blockBelong;
  auto CONTAIN_TAG = [&](int reqType) {
    auto p = irg->getActiveBlock();
    for (auto &s : p->blockType) {
      if (s.blockType == reqType)
        return true;
    }
    return false;
  };
  auto FORALL_CHAN = [&](std::function<void(int)> f) {
    for (int i = 0; i < irg->getQuads(); i++)
      f(i);
  };
  using MASK_TYPE = LLVM::SpVcLLVMTypeInt32;
  auto tpZero = irg->addIr(std::make_unique<MASK_TYPE>());
  auto MAKE_MASK = [&]() {
    auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
    return q;
  };

  auto MAKE_LABEL = [&]() {
    auto q = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMLabelName>(irg->allocateLlvmVarName()),
        irg->getActiveBlock());
    return q;
  };

  auto MAKE_MASK_ALLOC = [&](LLVM::SpVcLLVMArgument *q) {
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto p = irg->addIrF(std::make_unique<LLVM::SpVcLLVMIns_Alloca>(q, tp),
                         irg->getActiveBlock());
  };

  auto MAKE_MASK_INIT = [&](LLVM::SpVcLLVMArgument *q) {
    auto immOne =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto sq = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, immOne));
    auto p =
        irg->addIrF(std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(q, sq),
                    irg->getActiveBlock());
  };

  auto MAKE_MASK_AND = [&](LLVM::SpVcLLVMArgument *dest,
                           LLVM::SpVcLLVMArgument *a,
                           LLVM::SpVcLLVMArgument *b) {
    auto rDest = MAKE_MASK();
    auto rA = MAKE_MASK();
    auto rB = MAKE_MASK();
    // auto p = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_Load>(rDest,
    // dest), irg->getActiveBlock());
    auto q =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rA, a),
                    irg->getActiveBlock());
    auto s =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rB, b),
                    irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(rDest, rA, rB, "and"),
        irg->getActiveBlock());
    auto u = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(dest, rDest),
        irg->getActiveBlock());
  };
  auto MAKE_MASK_AND_TO_REG = [&](LLVM::SpVcLLVMArgument *dest,
                                  LLVM::SpVcLLVMArgument *a,
                                  LLVM::SpVcLLVMArgument *b) {
    auto rA = MAKE_MASK();
    auto rB = MAKE_MASK();
    auto q = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_Load>(rA, a),
                         irg->getActiveBlock());
    auto s =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rB, b),
                    irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(dest, rA, rB, "and"),
        irg->getActiveBlock());
  };
  auto MAKE_MASK_OR_REG = [&](LLVM::SpVcLLVMArgument *dest,
                              LLVM::SpVcLLVMArgument *a,
                              LLVM::SpVcLLVMArgument *b) {
    auto rB = MAKE_MASK();
    auto s = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_Load>(rB, b),
                         irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(dest, a, rB, "or"),
        irg->getActiveBlock());
  };
  auto MASK_ACT_LAST_PREV = [&](SpVcVMExecutionMask &v, int x) {
    if (v.activeExecMask.size() < 2) {
      EMIT_ERROR("Invalid active mask");
    }
    return v.activeExecMask[v.activeExecMask.size() - 2]->llvmVarName[x].arg;
  };
  auto sp = curB->stackBelong;

  if (CONTAIN_TAG(SPVC_BLOCK_LOOP_HEADER)) { // Indicates a loop break section
    // Write %1 to MASK_BR
    FORALL_CHAN([&](int x) {
      auto cond = GET_PARAM(0);
      auto mask = MASK_BREAK(sp->masks, x);
      auto condArg = cond->llvmVarName[x].arg;

      auto p = irg->addIrF(
          std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(mask, condArg),
          irg->getActiveBlock());
    });
    // AND MASK_BR_LAST with MASK_ACT_LAST
    FORALL_CHAN([&](int x) {
      auto mask = MASK_BREAK(sp->masks, x);
      auto act = MASK_ACT_LAST(sp->masks, x);
      MAKE_MASK_AND(act, mask, act);
    });

    // if all chan's MASK_ACT_LAST is zero, jump to merge
    // elsewise, jump to body
    LLVM::SpVcLLVMArgument *maskOrToken[4] = {MAKE_MASK(), MAKE_MASK(),
                                              MAKE_MASK(), MAKE_MASK()};
    FORALL_CHAN([&](int x) {
      if (x == 0) {
        MAKE_MASK_AND_TO_REG(maskOrToken[x], MASK_ACT_LAST(sp->masks, x),
                             MASK_ACT_LAST(sp->masks, x));
      } else {
        MAKE_MASK_OR_REG(maskOrToken[x], maskOrToken[x - 1],
                         MASK_ACT_LAST(sp->masks, x));
      }
    });
    // Compare maskOrToken[3] with 0
    auto immZero =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(0));
    auto immZeroArg =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpZero, immZero));
    auto cmpRetTp = irg->addIr(std::make_unique<LLVM::SpVcLLVMTypeBool>());
    auto cmpRetName =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
            irg->allocateLlvmVarName()));
    auto cmpRet = irg->addIr(
        std::make_unique<LLVM::SpVcLLVMArgument>(cmpRetTp, cmpRetName));
    auto cmp = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                               cmpRet, maskOrToken[3], immZeroArg, "icmp eq"),
                           irg->getActiveBlock());

    auto trueParam = GET_PARAM(1);
    auto falseParam = GET_PARAM(2);
    auto trueBlock = trueParam->blockBelong;
    auto falseBlock = falseParam->blockBelong;
    irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_BrCond>(
                    cmpRet, falseBlock->llvmLabel, trueBlock->llvmLabel),
                irg->getActiveBlock());

  } else if (CONTAIN_TAG(SPVC_BLOCK_LOOP_BREAK)) {
    EMIT_ERROR("Invalid loop break")
  } else if (CONTAIN_TAG(SPVC_BLOCK_LOOP_CONTINUE)) {
    EMIT_ERROR("It's valid, but not supported")
  }

  if (CONTAIN_TAG(SPVC_BLOCK_SELECTION_HEADER)) {
    // Write %1 to MASK_BR
    FORALL_CHAN([&](int x) {
      auto cond = GET_PARAM(0);
      auto mask = MASK_BR_LAST(sp->masks, x);
      auto condArg = cond->llvmVarName[x].arg;
      auto p = irg->addIrF(
          std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(mask, condArg),
          irg->getActiveBlock());
    });
    // AND MASK_BR_LAST with MASK_ACT_LAST_PR
    FORALL_CHAN([&](int x) {
      auto mask = MASK_BR_LAST(sp->masks, x);
      auto act = MASK_ACT_LAST(sp->masks, x);
      auto actPrev = MASK_ACT_LAST_PREV(sp->masks, x);
      MAKE_MASK_AND(act, mask, actPrev);
    });
  }
}
CONV_PASS(OpFunction) {
  auto fcnt = ctx->funcCounter++;

  irg->pushNewStack();
  auto sp = irg->getActiveStack();
  sp->ifStackSize = 1;
  auto func = GET_PARAM(1)->funcBelong;

  // GenIR
  auto irFuncName = irg->allocateLlvmVarName();
  irg->addIrFx(
      std::make_unique<LLVM::SpVcLLVMIns_FunctionFragmentEntry>(irFuncName),
      func);
  irg->addIrFxTail(std::make_unique<LLVM::SpVcLLVMIns_FunctionEnd>(), func);
  ctx->binds.mainFunction = irFuncName;

  // Init func
  auto FORALL_CHAN = [&](std::function<void(int)> f) {
    for (int i = 0; i < irg->getQuads(); i++)
      f(i);
  };
  using MASK_TYPE = LLVM::SpVcLLVMTypeInt32;
  auto MAKE_MASK = [&]() {
    auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
    return q;
  };
  auto MAKE_MASK_ALLOC = [&](LLVM::SpVcLLVMArgument *q) {
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto p =
        irg->addIrFx(std::make_unique<LLVM::SpVcLLVMIns_Alloca>(q, tp), func);
  };
  auto MAKE_MASK_INIT_LOCAL = [&](LLVM::SpVcLLVMArgument *q) {
    auto immOne =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto sq = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, immOne));
    auto p = irg->addIrFx(
        std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(q, sq), func);
  };

  sp->masks.execMask = irg->createExecutionMaskVar();
  sp->masks.returnMask = irg->createExecutionMaskVar();
  sp->masks.breakMask = irg->createExecutionMaskVar();
  sp->masks.continueMask = irg->createExecutionMaskVar();
  sp->masks.activeExecMask.push_back(irg->createExecutionMaskVar());
  sp->masks.branchMask.push_back(irg->createExecutionMaskVar());
  func->returnMask = irg->createExecutionMaskVar();

  FORALL_CHAN([&](int x) {
    MASK_EXEC(sp->masks, x) = MAKE_MASK();
    MAKE_MASK_ALLOC(MASK_EXEC(sp->masks, x));
    MASK_BREAK(sp->masks, x) = MAKE_MASK();
    MAKE_MASK_ALLOC(MASK_BREAK(sp->masks, x));
    MASK_CONT(sp->masks, x) = MAKE_MASK();
    MAKE_MASK_ALLOC(MASK_CONT(sp->masks, x));
    MASK_ACT_LAST(sp->masks, x) = MAKE_MASK();
    MAKE_MASK_ALLOC(MASK_ACT_LAST(sp->masks, x));
    MASK_BR_LAST(sp->masks, x) = MAKE_MASK();
    MAKE_MASK_ALLOC(MASK_BR_LAST(sp->masks, x));

    func->returnMask->llvmVarName[x].arg = MAKE_MASK();
    MAKE_MASK_ALLOC(func->returnMask->llvmVarName[x].arg);
    MAKE_MASK_INIT_LOCAL(func->returnMask->llvmVarName[x].arg);

    MAKE_MASK_INIT_LOCAL(MASK_EXEC(sp->masks, x));
    MAKE_MASK_INIT_LOCAL(MASK_BREAK(sp->masks, x));
    MAKE_MASK_INIT_LOCAL(MASK_CONT(sp->masks, x));
    MAKE_MASK_INIT_LOCAL(MASK_ACT_LAST(sp->masks, x));
    MAKE_MASK_INIT_LOCAL(MASK_BR_LAST(sp->masks, x));
  });
}

void convPassMathBinaryGeneral(int pc, std::vector<uint32_t> params,
                               SpVcVMGeneratorContext *ctx,
                               SpVcQuadGroupedIRGenerator *irg,
                               std::string op) {
  auto dest = GET_PARAM_SCALAR(1);
  auto opA = GET_PARAM_SCALAR(2);
  auto opB = GET_PARAM_SCALAR(3);
  foreachInvo(
      [&](int c) {
        auto destImmRes = spirvCreateVarWithSameType(dest, c, irg);
        auto opAImm = spirvImmediateLoad(opA, c, irg);
        auto opBImm = spirvImmediateLoad(opB, c, irg);
        auto res = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                                   destImmRes, opAImm, opBImm, op),
                               irg->getActiveBlock());
        spirvImmediateMaskedStore(dest, c, destImmRes, irg);
      },
      irg);
}
void convPassMathUnaryGeneral(int pc, std::vector<uint32_t> params,
                              SpVcVMGeneratorContext *ctx,
                              SpVcQuadGroupedIRGenerator *irg, std::string op) {
  auto dest = GET_PARAM_SCALAR(1);
  auto opA = GET_PARAM_SCALAR(2);
  foreachInvo(
      [&](int c) {
        auto destImmRes = spirvCreateVarWithSameType(dest, c, irg);
        auto opAImm = spirvImmediateLoad(opA, c, irg);
        auto res = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathUnary>(
                                   destImmRes, opAImm, op),
                               irg->getActiveBlock());
        spirvImmediateMaskedStore(dest, c, destImmRes, irg);
      },
      irg);
}

void convPassMathBinaryInteger(int pc, std::vector<uint32_t> params,
                               SpVcVMGeneratorContext *ctx,
                               SpVcQuadGroupedIRGenerator *irg,
                               std::string op) {
  auto dest = GET_PARAM_SCALAR(1);
  auto opA = GET_PARAM_SCALAR(2);
  auto opB = GET_PARAM_SCALAR(3);
  bool toUnsigned = false;
  if (GET_PARAM(2)->tpRef->tp->isUnsigned) {
    toUnsigned = true;
  }
  if (GET_PARAM(3)->tpRef->tp->isUnsigned) {
    toUnsigned = true;
  }
  if (toUnsigned) {
    op += " nsw";
  }
  foreachInvo(
      [&](int c) {
        auto destImmRes = spirvCreateVarWithSameType(dest, c, irg);
        auto opAImm = spirvImmediateLoad(opA, c, irg);
        auto opBImm = spirvImmediateLoad(opB, c, irg);
        auto res = irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                                   destImmRes, opAImm, opBImm, op),
                               irg->getActiveBlock());
        spirvImmediateMaskedStore(dest, c, destImmRes, irg);
      },
      irg);
}
CONV_PASS(OpFNegate) { convPassMathUnaryGeneral(pc, params, ctx, irg, "fneg"); }

CONV_PASS(OpFAdd) { convPassMathBinaryGeneral(pc, params, ctx, irg, "fadd"); }
CONV_PASS(OpFSub) { convPassMathBinaryGeneral(pc, params, ctx, irg, "fsub"); }
CONV_PASS(OpFMul) { convPassMathBinaryGeneral(pc, params, ctx, irg, "fmul"); }
CONV_PASS(OpFDiv) { convPassMathBinaryGeneral(pc, params, ctx, irg, "fdiv"); }

CONV_PASS(OpIAdd) { convPassMathBinaryInteger(pc, params, ctx, irg, "add"); }
CONV_PASS(OpISub) { convPassMathBinaryInteger(pc, params, ctx, irg, "sub"); }
CONV_PASS(OpIMul) { convPassMathBinaryInteger(pc, params, ctx, irg, "mul"); }

CONV_PASS(OpFOrdLessThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp olt");
}
CONV_PASS(OpFOrdGreaterThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp ogt");
}
CONV_PASS(OpFOrdLessThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp ole");
}
CONV_PASS(OpFOrdGreaterThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp oge");
}
CONV_PASS(OpFOrdEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp oeq");
}
CONV_PASS(OpFOrdNotEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp one");
}
CONV_PASS(OpFUnordLessThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp ult");
}
CONV_PASS(OpFUnordGreaterThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp ugt");
}
CONV_PASS(OpFUnordLessThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp ule");
}
CONV_PASS(OpFUnordGreaterThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp uge");
}
CONV_PASS(OpFUnordEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp ueq");
}
CONV_PASS(OpFUnordNotEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "fcmp une");
}

CONV_PASS(OpSLessThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp slt");
}
CONV_PASS(OpSGreaterThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp sgt");
}
CONV_PASS(OpSLessThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp sle");
}
CONV_PASS(OpSGreaterThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp sge");
}
CONV_PASS(OpULessThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp ult");
}
CONV_PASS(OpUGreaterThan) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp ugt");
}
CONV_PASS(OpULessThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp ule");
}
CONV_PASS(OpUGreaterThanEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp uge");
}
CONV_PASS(OpIEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp eq");
}
CONV_PASS(OpINotEqual) {
  convPassMathBinaryGeneral(pc, params, ctx, irg, "icmp ne");
}

CONV_PASS(OpVariable) {
  auto tgt = GET_PARAM(1);
  auto tpLlvm = tgt->tpRef->tp->llvmType;
  if (tgt->isAllQuad) {
    auto arg = tgt->llvmVarName[0].arg;
    if (irg->getActiveBlock() == nullptr) {
      // Is global
      auto allocaIns =
          irg->addIrG(std::make_unique<LLVM::SpVcLLVMIns_GlobalVariable>(arg));
    } else {
      auto allocaIns =
          irg->addIrFx(std::make_unique<LLVM::SpVcLLVMIns_Alloca>(arg, tpLlvm),
                       tgt->blockBelong->funcBelong);
    }

  } else {
    foreachInvo(
        [&](int x) {
          auto arg = tgt->llvmVarName[x].arg;
          if (irg->getActiveBlock() == nullptr) {
            // Is global
            auto allocaIns = irg->addIrG(
                std::make_unique<LLVM::SpVcLLVMIns_GlobalVariable>(arg));
          } else {
            auto allocaIns = irg->addIrFx(
                std::make_unique<LLVM::SpVcLLVMIns_Alloca>(arg, tpLlvm),
                tgt->blockBelong->funcBelong);
          }
        },
        irg);
  }
}

CONV_PASS(OpLoad) {
  auto tgt = GET_PARAM(1);
  auto tgtSc = GET_PARAM_SCALAR(1);
  auto src = GET_PARAM(2);
  auto tpLlvm = tgt->tpRef->tp->llvmType;
  foreachInvo(
      [&](int x) {
        auto arg = tgt->llvmVarName[x].arg;
        auto srcArg = src->llvmVarName[x].arg;
        auto ld = spirvImmediateLoad(src->id, x, irg);
        spirvImmediateMaskedStore(tgt->id, x, ld, irg);
      },
      irg);
}

CONV_PASS(OpStore) {
  auto tgt = GET_PARAM(0);
  auto src = GET_PARAM(1);
  if (tgt->flag & SPVC_VARIABLE_ACCESS_CHAIN) {
    EMIT_ERROR("Unsupported chain access");
  }
  foreachInvo(
      [&](int x) {
        auto arg = tgt->llvmVarName[x].arg;
        auto srcArg = src->llvmVarName[x].arg;
        auto ld = spirvImmediateLoad(src->id, x, irg);
        spirvImmediateMaskedStore(tgt->id, x, ld, irg);
      },
      irg);
}

LLVM::SpVcLLVMArgument *
spirvCompositeConstruct(int invo, int &p, const std::vector<uint32_t> &params,
                        SpVcVMTypeDescriptor *tp,
                        SpVcQuadGroupedIRGenerator *irg) {
  LLVM::SpVcLLVMArgument *last = nullptr;
  auto b = irg->getActiveBlock();
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_VECTOR) {
    std::vector<LLVM::SpVcLLVMArgument *> targs;

    for (int i = 0; i < tp->size; i++) {
      auto loadIns = spirvImmediateLoad(params[p + i], invo, irg);
      targs.push_back(loadIns);
      auto rp = spirvCreateVarWithType(tp, irg);
      if (i == 0) {
        irg->addIrB(
            std::make_unique<
                LLVM::SpVcLLVMIns_InsertElementWithConstantIndexUndefInit>(
                rp, loadIns, i),
            b);
      } else {
        irg->addIrB(
            std::make_unique<LLVM::SpVcLLVMIns_InsertElementWithConstantIndex>(
                rp, last, loadIns, i),
            b);
      }
      last = rp;
    }
  } else if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_MATRIX) {
    EMIT_WARN("Matrix layout ignored");
    int cnt = 0;
    for (int i = 0; i < tp->size; i++) {
      auto loadIns = spirvImmediateLoad(params[p + i], invo, irg);
      auto vecSize = irg->getVariableSafe(params[p + i])->tpRef->tp->size;
      auto vecBaseTp =
          irg->getVariableSafe(params[p + i])->tpRef->tp->children[0]->llvmType;

      std::vector<LLVM::SpVcLLVMArgument *> regArgs;
      for (int j = 0; j < vecSize; j++) {
        auto regName =
            irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                irg->allocateLlvmVarName()));
        auto regArg = irg->addIr(
            std::make_unique<LLVM::SpVcLLVMArgument>(vecBaseTp, regName));
        regArgs.push_back(regArg);
        irg->addIrB(
            std::make_unique<LLVM::SpVcLLVMIns_ExtractElementWithConstantIndex>(
                regArgs[j], loadIns, j),
            b);
      }
      for (int j = 0; j < vecSize; j++) {
        auto rp = spirvCreateVarWithType(tp, irg);
        if (cnt == 0) {
          irg->addIrB(
              std::make_unique<
                  LLVM::SpVcLLVMIns_InsertElementWithConstantIndexUndefInit>(
                  rp, regArgs[j], cnt),
              b);
        } else {
          irg->addIrB(std::make_unique<
                          LLVM::SpVcLLVMIns_InsertElementWithConstantIndex>(
                          rp, last, regArgs[j], cnt),
                      b);
        }
        cnt++;
        last = rp;
      }
    }
  } else {
    EMIT_ERROR("Unsupported type");
  }
  return last;
}
CONV_PASS(OpCompositeConstruct) {
  auto tgt = GET_PARAM(1);
  auto tp = tgt->tpRef->tp.get();
  auto p = 2;
  foreachInvo(
      [&](int x) {
        auto arg = spirvCompositeConstruct(x, p, params, tp, irg);
        spirvImmediateMaskedStore(tgt->id, x, arg, irg);
      },
      irg);
}

LLVM::SpVcLLVMArgument *
spirvCompositeExtract(int invo, int &p, const std::vector<uint32_t> &params,
                      SpVcVMTypeDescriptor *tp, int tgtId,
                      SpVcQuadGroupedIRGenerator *irg) {
  auto tgtType = irg->getVariableSafe(tgtId)->tpRef->tp.get();
  if (tgtType->type == SpVcVMTypeEnum::SPVC_TYPE_VECTOR) {
    auto vecReg = spirvImmediateLoad(tgtId, invo, irg);
    auto tpReg = irg->getVariableSafe(tgtId)->tpRef->tp->children[0]->llvmType;
    auto idx = params[p];
    auto regName = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto regArg =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpReg, regName));
    irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_ExtractElementWithConstantIndex>(
            regArg, vecReg, idx),
        irg->getActiveBlock());
    return regArg;
  }
  EMIT_ERROR("Unsupported type: composite extract");
}

CONV_PASS(OpCompositeExtract) {
  auto tgt = GET_PARAM(1);
  auto tp = tgt->tpRef->tp.get();
  auto p = 3;
  foreachInvo(
      [&](int x) {
        auto arg =
            spirvCompositeExtract(x, p, params, tp, GET_PARAM(2)->id, irg);
        spirvImmediateMaskedStore(tgt->id, x, arg, irg);
      },
      irg);
}

CONV_PASS(OpDot) {
  auto src1 = GET_PARAM(2);
  auto src2 = GET_PARAM(3);
  foreachInvo(
      [&](int x) {
        auto reg1 = spirvImmediateLoad(src1->id, x, irg);
        auto reg2 = spirvImmediateLoad(src2->id, x, irg);
        auto elType =
            irg->getVariableSafe(src1->id)->tpRef->tp->children[0]->llvmType;
        std::vector<LLVM::SpVcLLVMArgument *> args;
        for (int i = 0; i < irg->getVariableSafe(src1->id)->tpRef->tp->size;
             i++) {
          auto regName =
              irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                  irg->allocateLlvmVarName()));
          auto regArg = irg->addIr(
              std::make_unique<LLVM::SpVcLLVMArgument>(elType, regName));
          args.push_back(regArg);
          irg->addIrB(std::make_unique<
                          LLVM::SpVcLLVMIns_ExtractElementWithConstantIndex>(
                          args[i], reg1, i),
                      irg->getActiveBlock());
        }
        // Sum up
        std::vector<LLVM::SpVcLLVMArgument *> largs;
        for (int i = 0; i < irg->getVariableSafe(src1->id)->tpRef->tp->size - 1;
             i++) {
          if (i == 0) {
            auto regName =
                irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                    irg->allocateLlvmVarName()));
            auto regArg = irg->addIr(
                std::make_unique<LLVM::SpVcLLVMArgument>(elType, regName));
            irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                            regArg, args[i], args[i + 1], "fmul"),
                        irg->getActiveBlock());
            largs.push_back(regArg);
          } else {
            auto regName2 =
                irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                    irg->allocateLlvmVarName()));
            auto regArg2 = irg->addIr(
                std::make_unique<LLVM::SpVcLLVMArgument>(elType, regName2));
            auto regName =
                irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                    irg->allocateLlvmVarName()));
            auto regArg = irg->addIr(
                std::make_unique<LLVM::SpVcLLVMArgument>(elType, regName));
            irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                            regArg2, args[i], args[i + 1], "fmul"),
                        irg->getActiveBlock());
            irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                            regArg, largs.back(), regArg2, "fadd"),
                        irg->getActiveBlock());
            largs.push_back(regArg);
          }
        }
        spirvImmediateMaskedStore(GET_PARAM(1)->id, x, largs.back(), irg);
      },
      irg);
}

CONV_PASS(OpVectorShuffle) {
  auto tgt = GET_PARAM(1);
  auto src1 = GET_PARAM(2);
  auto src2 = GET_PARAM(3);
  // masks = 4,5,6,7 etc
  foreachInvo(
      [&](int x) {
        auto src1Reg = spirvImmediateLoad(src1->id, x, irg);
        auto src2Reg = spirvImmediateLoad(src2->id, x, irg);
        auto tpX =
            irg->getVariableSafe(src1->id)->tpRef->tp->children[0]->llvmType;
        auto tpP = irg->getVariableSafe(src1->id)->tpRef->tp->llvmType;

        std::vector<LLVM::SpVcLLVMArgument *> args;
        std::vector<LLVM::SpVcLLVMArgument *> ret;
        for (int i = 4; i < params.size(); i++) {
          auto regName =
              irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                  irg->allocateLlvmVarName()));
          auto regArg = irg->addIr(
              std::make_unique<LLVM::SpVcLLVMArgument>(tpX, regName));
          args.push_back(regArg);
          auto sInd = params[i];
          auto vecSize = irg->getVariableSafe(src1->id)->tpRef->tp->size;
          auto procVec = sInd >= vecSize ? src2Reg : src1Reg;
          auto procVecIdx = sInd >= vecSize ? sInd - vecSize : sInd;
          irg->addIrB(std::make_unique<
                          LLVM::SpVcLLVMIns_ExtractElementWithConstantIndex>(
                          regArg, procVec, procVecIdx),
                      irg->getActiveBlock());

          // Build ret
          auto retName =
              irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                  irg->allocateLlvmVarName()));
          auto retArg = irg->addIr(
              std::make_unique<LLVM::SpVcLLVMArgument>(tpP, retName));
          ret.push_back(retArg);
          if (i == 4) {
            irg->addIrB(
                std::make_unique<
                    LLVM::SpVcLLVMIns_InsertElementWithConstantIndexUndefInit>(
                    retArg, regArg, 0),
                irg->getActiveBlock());
          } else {
            irg->addIrB(std::make_unique<
                            LLVM::SpVcLLVMIns_InsertElementWithConstantIndex>(
                            retArg, ret[i - 5], regArg, i - 4),
                        irg->getActiveBlock());
          }
        }
      },
      irg);
}

CONV_PASS(OpExtInst) {
  auto reg = irg->getExtRegistry();
  std::vector<Spirv::SpvVMExtRegistryTypeIdentifier> ids;
  std::vector<int> comSize;

  using checkIdRet = std::pair<Spirv::SpvVMExtRegistryTypeIdentifier, int>;
  std::function<checkIdRet(SpVcVMTypeDescriptor *)> checkId =
      [&checkId](SpVcVMTypeDescriptor *desc) -> checkIdRet {
    if (desc->type == SpVcVMTypeEnum::SPVC_TYPE_VECTOR) {
      auto comSize = desc->size;
      auto p = checkId(desc->children[0]);
      if (p.second == -1) {
        return {Spirv::SpvVMExtRegistryTypeIdentifier::IFSP_EXTREG_TP_INT, -1};
      }
      return {p.first, comSize};
    } else if (desc->type == SpVcVMTypeEnum::SPVC_TYPE_INT32) {
      return {Spirv::SpvVMExtRegistryTypeIdentifier::IFSP_EXTREG_TP_INT, 1};
    } else if (desc->type == SpVcVMTypeEnum::SPVC_TYPE_FLOAT32) {
      return {Spirv::SpvVMExtRegistryTypeIdentifier::IFSP_EXTREG_TP_FLOAT, 1};
    } else {
      return {Spirv::SpvVMExtRegistryTypeIdentifier::IFSP_EXTREG_TP_INT, -1};
    }
  };

  for (int i = 4; i < params.size(); i++) {
    auto tpDesc = checkId(irg->getVariableSafe(params[i])->tpRef->tp.get());
    if (tpDesc.second == -1) {
      EMIT_ERROR("Unsupported type");
    }
    ids.push_back(tpDesc.first);
    comSize.push_back(tpDesc.second);
  }
  auto extName = irg->getVariableSafe(params[2])->name;
  auto extOp = GET_PARAM_SCALAR(3);
  auto funcName = reg->queryExternalFunc(extName, extOp, ids, comSize);

  foreachInvo(
      [&](int x) {
        std::vector<LLVM::SpVcLLVMArgument *> args;
        for (int i = 4; i < params.size(); i++) {
          auto arg = spirvImmediateLoad(params[i], x, irg);
          args.push_back(arg);
        }
        auto retVal = spirvCreateVarWithSameType(GET_PARAM_SCALAR(1), x, irg);
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_FunctionCallExtInst>(
                        retVal, funcName, args),
                    irg->getActiveBlock());
        spirvImmediateMaskedStore(GET_PARAM(1)->id, x, retVal, irg);
      },
      irg);
}

// Access chain
LLVM::SpVcLLVMArgument *accessChainLookup(LLVM::SpVcLLVMArgument *srcReg,
                                          int &p, std::vector<uint32_t> params,
                                          SpVcQuadGroupedIRGenerator *irg,
                                          SpVcVMTypeDescriptor *tp) {
  if (p == params.size())
    return srcReg;
  if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_VECTOR) {
    auto idx = irg->getVariableSafe(params[p])->constant->value[0];
    auto tpReg = tp->children[0]->llvmType;
    auto regName = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto regArg =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpReg, regName));
    irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_ExtractElementWithConstantIndex>(
            regArg, srcReg, idx),
        irg->getActiveBlock());
    return regArg;
  } else if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_STRUCT) {
    auto idx = irg->getVariableSafe(params[p])->constant->value[0];
    auto tpReg = tp->children[idx]->llvmType;
    auto regName = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto regArg =
        irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpReg, regName));
    irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_ExtractValueWithConstantIndex>(
            regArg, srcReg, idx),
        irg->getActiveBlock());
    return accessChainLookup(regArg, ++p, params, irg, tp->children[idx]);
  }
  EMIT_ERROR("Unsupported access chain");
}

CONV_PASS(OpAccessChain) {
  auto tgt = GET_PARAM(1);
  auto src = GET_PARAM(2);
  auto tgtTp = tgt->tpRef->tp.get();

  auto srcTp = src->tpRef->tp.get();
  foreachInvo(
      [&](int x) {
        auto p = 3;
        auto arg = accessChainLookup(spirvImmediateLoad(src->id, x, irg), p,
                                     params, irg, srcTp);
        spirvImmediateMaskedStore(tgt->id, x, arg, irg);
      },
      irg);
}

CONV_PASS(OpReturn) {
  // Fetch all in MASK_ACT_LAST, if it's one, set RET_MASK to zero
  using MASK_TYPE = LLVM::SpVcLLVMTypeInt32;
  auto MAKE_MASK = [&]() {
    auto p = irg->addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
        irg->allocateLlvmVarName()));
    auto tp = irg->addIr(std::make_unique<MASK_TYPE>());
    auto q = irg->addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, p));
    return q;
  };
  auto MAKE_MASK_AND = [&](LLVM::SpVcLLVMArgument *dest,
                           LLVM::SpVcLLVMArgument *a,
                           LLVM::SpVcLLVMArgument *b) {
    auto rDest = MAKE_MASK();
    auto rA = MAKE_MASK();
    auto rB = MAKE_MASK();
    // auto p =
    // irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rDest,
    // dest), irg->getActiveBlock());
    auto q =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rA, a),
                    irg->getActiveBlock());
    auto s =
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rB, b),
                    irg->getActiveBlock());
    auto t = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(rDest, rA, rB, "and"),
        irg->getActiveBlock());
    auto u = irg->addIrB(
        std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(dest, rDest),
        irg->getActiveBlock());
  };

  foreachInvo(
      [&](int x) {
        // load MASK_ACT_LAST
        auto curB = irg->getActiveBlock();
        auto rA = MAKE_MASK();
        auto maskActLast = MASK_ACT_LAST(curB->stackBelong->masks, x);
        irg->addIrB(
            std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rA, maskActLast),
            curB);
        auto maskRet = curB->funcBelong->returnMask->llvmVarName[x].arg;
        auto rRet = MAKE_MASK();
        irg->addIrB(
            std::make_unique<LLVM::SpVcLLVMIns_LoadForcedPtr>(rRet, maskRet),
            curB);

        auto rDest = MAKE_MASK();
        auto allOneImm =
            irg->addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(1));
        auto allOneImmType = irg->addIr(std::make_unique<MASK_TYPE>());
        auto allOneArg = irg->addIr(
            std::make_unique<LLVM::SpVcLLVMArgument>(allOneImmType, allOneImm));
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                        rDest, allOneArg, rA, "sub"),
                    irg->getActiveBlock());

        auto rDest2 = MAKE_MASK();
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                        rDest2, rRet, rDest, "and"),
                    irg->getActiveBlock());
        irg->addIrB(
            std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(maskRet, rDest2),
            irg->getActiveBlock());

        auto rAp = MAKE_MASK();
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(rAp, rDest2,
                                                                   rA, "and"),
                    irg->getActiveBlock());
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_StoreForcedPtr>(
                        maskActLast, rAp),
                    irg->getActiveBlock());
      },
      irg);
  // TODO: Return
  irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_ReturnVoid>(),
              irg->getActiveBlock());
}

CONV_PASS(OpVectorTimesScalar) {
  auto src1 = GET_PARAM(2); // Vec
  auto src2 = GET_PARAM(3); // Scalar
  foreachInvo(
      [&](int x) {
        auto vecReg = spirvImmediateLoad(src1->id, x, irg);
        auto scalarReg = spirvImmediateLoad(src2->id, x, irg);
        auto tpX =
            irg->getVariableSafe(src1->id)->tpRef->tp->children[0]->llvmType;
        auto tpP = irg->getVariableSafe(src1->id)->tpRef->tp->llvmType;
        auto vectorSize = irg->getVariableSafe(src1->id)->tpRef->tp->size;
        // Scalar to vector
        LLVM::SpVcLLVMArgument *last = nullptr;
        for (int i = 0; i < vectorSize; i++) {
          auto rp = spirvCreateVarWithType(src1->tpRef->tp.get(), irg);
          if (i == 0)
            irg->addIrB(
                std::make_unique<
                    LLVM::SpVcLLVMIns_InsertElementWithConstantIndexUndefInit>(
                    rp, scalarReg, 0),
                irg->getActiveBlock());
          else
            irg->addIrB(std::make_unique<
                            LLVM::SpVcLLVMIns_InsertElementWithConstantIndex>(
                            rp, last, scalarReg, i),
                        irg->getActiveBlock());
          last = rp;
        }
        auto regArg = spirvCreateVarWithType(src1->tpRef->tp.get(), irg);
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                        regArg, vecReg, last, "fmul"),
                    irg->getActiveBlock());
        spirvImmediateMaskedStore(GET_PARAM_SCALAR(1), x, regArg, irg);
      },
      irg);
}

CONV_PASS(OpDPDx) {
  auto src = GET_PARAM(2);
  // Quad Organization:
  // 0 1
  // 2 3
  static_assert(SpVcQuadSize == 4, "Quad size should be 4");
  foreachInvo(
      [&](int x) {
        int lowerId = x & 0x2;
        int upperId = lowerId + 1;
        auto lowerReg = spirvImmediateLoad(src->id, lowerId, irg);
        auto upperReg = spirvImmediateLoad(src->id, upperId, irg);
        // fsub instruction
        auto resultType = GET_PARAM(0)->tp.get();
        auto resultReg = spirvCreateVarWithType(resultType, irg);
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                        resultReg, upperReg, lowerReg, "fsub"),
                    irg->getActiveBlock());
        spirvImmediateMaskedStore(GET_PARAM(1)->id, x, resultReg, irg);
      },
      irg);
}
CONV_PASS(OpDPDy) {
  auto src = GET_PARAM(2);
  // Quad Organization:
  // 0 1
  // 2 3
  static_assert(SpVcQuadSize == 4, "Quad size should be 4");
  foreachInvo(
      [&](int x) {
        int lowerId = x & 0x1;
        int upperId = lowerId + 2;
        auto lowerReg = spirvImmediateLoad(src->id, lowerId, irg);
        auto upperReg = spirvImmediateLoad(src->id, upperId, irg);
        // fsub instruction
        auto resultType = GET_PARAM(0)->tp.get();
        auto resultReg = spirvCreateVarWithType(resultType, irg);
        irg->addIrB(std::make_unique<LLVM::SpVcLLVMIns_MathBinary>(
                        resultReg, upperReg, lowerReg, "fsub"),
                    irg->getActiveBlock());
        spirvImmediateMaskedStore(GET_PARAM(1)->id, x, resultReg, irg);
      },
      irg);
}
} // namespace ConversionPass

SpVcVMGenBlock *SpVcQuadGroupedIRGenerator::getActiveBlock() {
  if (mCtx->blockStack.size() == 0) {
    return nullptr;
  }
  return mCtx->blockStack.back();
}

void SpVcQuadGroupedIRGenerator::pushActiveBlock(SpVcVMGenBlock *block) {
  if (block == nullptr) {
    EMIT_ERROR_CORE("Block is null");
  }
  mCtx->blockStack.push_back(block);
}

SpVcVMGenStack *SpVcQuadGroupedIRGenerator::getActiveStack() {
  if (mCtx->structStack.empty()) {
    ifritError("Wrong");
  }
  return mCtx->structStack.back();
}

void SpVcQuadGroupedIRGenerator::pushNewStack() {
  auto uv = std::make_unique<SpVcVMGenStack>();
  auto ptruv = uv.get();
  mCtx->genstack.push_back(std::move(uv));
  mCtx->structStack.push_back(ptruv);
}

void SpVcQuadGroupedIRGenerator::popNewStack() { mCtx->structStack.pop_back(); }

SpVcVMGenVariable *SpVcQuadGroupedIRGenerator::createExecutionMaskVar() {
  auto maskVar = getVariableSafe(--curVarMask);
  maskVar->tp = std::make_unique<SpVcVMTypeDescriptor>();
  maskVar->tp->type = SpVcVMTypeEnum::SPVC_TYPE_INT32;
  maskVar->flag |= SPVC_VARIABLE_VAR;
  maskVar->name = "execMask";
  maskVar->blockBelong = getActiveBlock();
  maskVar->llvmVarName.resize(getQuads());
  return maskVar;
}

SpVcVMGenVariable *SpVcQuadGroupedIRGenerator::getVariableSafe(uint32_t id) {
  if (mCtx->v.count(id) == 0 || mCtx->v[id].id == -1) {
    mCtx->v[id] = SpVcVMGenVariable();
    mCtx->v[id].id = id;
  }
  return &mCtx->v[id];
}

std::string SpVcQuadGroupedIRGenerator::getParsingProgress() {
  return "[Pass " + std::to_string(curStage) + ", Line " +
         std::to_string(curPc) + "]";
}
void SpVcQuadGroupedIRGenerator::setCurrentProgCounter(int pc) { curPc = pc; }

void SpVcQuadGroupedIRGenerator::setCurrentPass(SpVcQuadGroupedIRStage stage) {
  curStage = stage;
}

std::string SpVcQuadGroupedIRGenerator::allocateLlvmVarName() {
  curVar++;
  return "a" + std::to_string(curVar);
}

void SpVcQuadGroupedIRGenerator::bindBytecode(SpVcSpirBytecode *bytecode,
                                              SpVcVMGeneratorContext *context) {
  mRaw = bytecode;
  mCtx = context;
}

void SpVcQuadGroupedIRGenerator::performDefinitionPassRegister() {
  mUniversalDefinitionPassHandler = GET_DEF_PASS(DefOpUniversal);

  mDefinitionPassHandlers[spv::Op::OpCapability] = GET_DEF_PASS(OpCapability);
  mDefinitionPassHandlers[spv::Op::OpExtInstImport] =
      GET_DEF_PASS(OpExtInstImport);
  mDefinitionPassHandlers[spv::Op::OpMemoryModel] = GET_DEF_PASS(OpMemoryModel);
  mDefinitionPassHandlers[spv::Op::OpEntryPoint] = GET_DEF_PASS(OpEntryPoint);
  mDefinitionPassHandlers[spv::Op::OpExecutionMode] =
      GET_DEF_PASS(OpExecutionMode);
  mDefinitionPassHandlers[spv::Op::OpSource] = GET_DEF_PASS(OpSource);
  mDefinitionPassHandlers[spv::Op::OpSourceExtension] =
      GET_DEF_PASS(OpSourceExtension);
  mDefinitionPassHandlers[spv::Op::OpName] = GET_DEF_PASS(OpName);
  mDefinitionPassHandlers[spv::Op::OpMemberName] = GET_DEF_PASS(OpMemberName);

  mDefinitionPassHandlers[spv::Op::OpTypeFloat] = GET_DEF_PASS(OpTypeFloat);
  mDefinitionPassHandlers[spv::Op::OpTypeInt] = GET_DEF_PASS(OpTypeInt);
  mDefinitionPassHandlers[spv::Op::OpTypePointer] = GET_DEF_PASS(OpTypePointer);
  mDefinitionPassHandlers[spv::Op::OpTypeVector] = GET_DEF_PASS(OpTypeVector);
  mDefinitionPassHandlers[spv::Op::OpTypeBool] = GET_DEF_PASS(OpTypeBool);
  mDefinitionPassHandlers[spv::Op::OpTypeArray] = GET_DEF_PASS(OpTypeArray);
  mDefinitionPassHandlers[spv::Op::OpTypeVoid] = GET_DEF_PASS(OpTypeVoid);
  mDefinitionPassHandlers[spv::Op::OpTypeStruct] = GET_DEF_PASS(OpTypeStruct);
  mDefinitionPassHandlers[spv::Op::OpTypeFunction] =
      GET_DEF_PASS(OpTypeFunction);
  mDefinitionPassHandlers[spv::Op::OpTypeImage] = GET_DEF_PASS(OpTypeImage);
  mDefinitionPassHandlers[spv::Op::OpTypeSampler] = GET_DEF_PASS(OpTypeSampler);
  mDefinitionPassHandlers[spv::Op::OpTypeSampledImage] =
      GET_DEF_PASS(OpTypeSampledImage);
  mDefinitionPassHandlers[spv::Op::OpTypeMatrix] = GET_DEF_PASS(OpTypeMatrix);

  mDefinitionPassHandlers[spv::Op::OpConstant] = GET_DEF_PASS(OpConstant);
  mDefinitionPassHandlers[spv::Op::OpConstantComposite] =
      GET_DEF_PASS(OpConstantComposite);

  mDefinitionPassHandlers[spv::Op::OpVariable] = GET_DEF_PASS(OpVariable);
  mDefinitionPassHandlers[spv::Op::OpAccessChain] = GET_DEF_PASS(OpAccessChain);

  mDefinitionPassHandlers[spv::Op::OpDecorate] = GET_DEF_PASS(OpDecorate);
  mDefinitionPassHandlers[spv::Op::OpMemberDecorate] =
      GET_DEF_PASS(OpMemberDecorate);

  mDefinitionPassHandlers[spv::Op::OpSelectionMerge] =
      GET_DEF_PASS(OpSelectionMerge);
  mDefinitionPassHandlers[spv::Op::OpLoopMerge] = GET_DEF_PASS(OpLoopMerge);
  mDefinitionPassHandlers[spv::Op::OpBranchConditional] =
      GET_DEF_PASS(OpBranchConditional);
  mDefinitionPassHandlers[spv::Op::OpBranch] = GET_DEF_PASS(OpBranch);
  mDefinitionPassHandlers[spv::Op::OpSwitch] = GET_DEF_PASS(OpSwitch);
  mDefinitionPassHandlers[spv::Op::OpLabel] = GET_DEF_PASS(OpLabel);

  mDefinitionPassHandlers[spv::Op::OpFunction] = GET_DEF_PASS(OpFunction);
  mDefinitionPassHandlers[spv::Op::OpFunctionEnd] = GET_DEF_PASS(OpFunctionEnd);

  mDefinitionPassHandlers[spv::Op::OpStore] = GET_DEF_PASS(DefOpIgnore);
  mDefinitionPassHandlers[spv::Op::OpReturn] = GET_DEF_PASS(OpReturn);
  mDefinitionPassHandlers[spv::Op::OpUnreachable] = GET_DEF_PASS(DefOpIgnore);
}

void SpVcQuadGroupedIRGenerator::performDataflowResolutionPassRegister() {
  // First element: result type (-2) None
  // Otherwise: -1 -> all
  auto DATA_DEP = [](int ret, std::vector<int> dep,
                     int vaDep) -> SpVcDataflowDependency {
    return {ret, std::vector<int>(dep), vaDep};
  };
  auto DATA_DEP_SPECIAL =
      [](SpVcDataflowDependencySpecial x) -> SpVcDataflowDependency {
    return {-1, {}, -1, x};
  };

  mArgumentDependency[spv::Op::OpCapability] = {};
  mArgumentDependency[spv::Op::OpExtInstImport] = {};
  mArgumentDependency[spv::Op::OpMemoryModel] = {};
  mArgumentDependency[spv::Op::OpEntryPoint] = {};
  mArgumentDependency[spv::Op::OpExecutionMode] = {};
  mArgumentDependency[spv::Op::OpSource] = {};
  mArgumentDependency[spv::Op::OpSourceExtension] = {};
  mArgumentDependency[spv::Op::OpName] = {};

  mArgumentDependency[spv::Op::OpTypeFloat] = DATA_DEP(0, {}, -1);
  mArgumentDependency[spv::Op::OpTypeInt] = DATA_DEP(0, {}, -1);
  mArgumentDependency[spv::Op::OpTypePointer] = DATA_DEP(0, {2}, -1);
  mArgumentDependency[spv::Op::OpTypeVector] = DATA_DEP(0, {1}, -1);
  mArgumentDependency[spv::Op::OpTypeBool] = DATA_DEP(0, {}, -1);
  mArgumentDependency[spv::Op::OpTypeArray] = DATA_DEP(0, {1}, -1);
  mArgumentDependency[spv::Op::OpTypeVoid] = DATA_DEP(0, {}, -1);
  mArgumentDependency[spv::Op::OpTypeStruct] = DATA_DEP(0, {}, 1);
  mArgumentDependency[spv::Op::OpTypeFunction] = DATA_DEP(0, {}, 1);
  mArgumentDependency[spv::Op::OpTypeImage] = DATA_DEP(0, {1}, -1);
  mArgumentDependency[spv::Op::OpTypeSampler] = DATA_DEP(0, {}, -1);
  mArgumentDependency[spv::Op::OpTypeSampledImage] = DATA_DEP(0, {}, 1);

  mArgumentDependency[spv::Op::OpConstant] = DATA_DEP(0, {1}, -1);
  mArgumentDependency[spv::Op::OpConstantComposite] = DATA_DEP(0, {}, 1);

  mArgumentDependency[spv::Op::OpVariable] = DATA_DEP(1, {0}, 3);

  mArgumentDependency[spv::Op::OpDecorate] = {};
  mArgumentDependency[spv::Op::OpMemberDecorate] = {};

  mArgumentDependency[spv::Op::OpSelectionMerge] = {};
  mArgumentDependency[spv::Op::OpLoopMerge] = {};
  mArgumentDependency[spv::Op::OpBranchConditional] = DATA_DEP(0, {}, -1);
  mArgumentDependency[spv::Op::OpBranch] = {};
  mArgumentDependency[spv::Op::OpSwitch] = DATA_DEP(0, {}, -1);
  mArgumentDependency[spv::Op::OpLabel] = {};

  mArgumentDependency[spv::Op::OpFunction] = {};
  mArgumentDependency[spv::Op::OpFunctionEnd] = {};

  mArgumentDependency[spv::Op::OpStore] = DATA_DEP(-1, {}, 0);
  mArgumentDependency[spv::Op::OpLoad] = DATA_DEP(1, {}, 0);
  mArgumentDependency[spv::Op::OpReturn] = {};
  mArgumentDependency[spv::Op::OpReturnValue] = DATA_DEP(-1, {0}, -1);

  mArgumentDependency[spv::Op::OpExtInst] = DATA_DEP(1, {0, 2}, 4);
  mArgumentDependency[spv::Op::OpFunctionCall] = DATA_DEP(1, {0}, 2);

  mArgumentDependency[spv::Op::OpImageSampleImplicitLod] =
      DATA_DEP(1, {0, 2, 3}, 5);
  mArgumentDependency[spv::Op::OpImageSampleExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 5}, 6);
  mArgumentDependency[spv::Op::OpImageSampleDrefImplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4}, 6);
  mArgumentDependency[spv::Op::OpImageSampleDrefExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4, 6}, 7);
  mArgumentDependency[spv::Op::OpImageSampleProjImplicitLod] =
      DATA_DEP(1, {0, 2, 3}, 5);
  mArgumentDependency[spv::Op::OpImageSampleProjExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 5}, 6);
  mArgumentDependency[spv::Op::OpImageSampleProjDrefImplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4}, 6);
  mArgumentDependency[spv::Op::OpImageSampleProjDrefExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4, 6}, 7);
  mArgumentDependency[spv::Op::OpImageFetch] = DATA_DEP(1, {0, 2, 3}, 5);
  mArgumentDependency[spv::Op::OpImageGather] = DATA_DEP(1, {0, 2, 3, 4}, 6);
  mArgumentDependency[spv::Op::OpImageDrefGather] =
      DATA_DEP(1, {0, 2, 3, 4, 5}, 7);
  mArgumentDependency[spv::Op::OpImageQuerySizeLod] =
      DATA_DEP(1, {0, 2, 3}, -1);
  mArgumentDependency[spv::Op::OpImageQuerySize] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpImageQueryLod] = DATA_DEP(1, {0, 2, 3}, -1);
  mArgumentDependency[spv::Op::OpImageQueryLevels] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpImageQuerySamples] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpImageSparseSampleImplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4}, 6);
  mArgumentDependency[spv::Op::OpImageSparseSampleExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4, 6}, 7);
  mArgumentDependency[spv::Op::OpImageSparseSampleDrefImplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4, 5}, 7);
  mArgumentDependency[spv::Op::OpImageSparseSampleDrefExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4, 5, 7}, 8);
  mArgumentDependency[spv::Op::OpImageSparseSampleProjImplicitLod] =
      DATA_DEP(1, {0, 2, 3}, 5);
  mArgumentDependency[spv::Op::OpImageSparseSampleProjExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 5}, 6);
  mArgumentDependency[spv::Op::OpImageSparseSampleProjDrefImplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4}, 6);
  mArgumentDependency[spv::Op::OpImageSparseSampleProjDrefExplicitLod] =
      DATA_DEP(1, {0, 2, 3, 4, 6}, 7);
  mArgumentDependency[spv::Op::OpImageSparseFetch] = DATA_DEP(1, {0, 2, 3}, 5);
  mArgumentDependency[spv::Op::OpImageSparseGather] =
      DATA_DEP(1, {0, 2, 3, 4}, 6);
  mArgumentDependency[spv::Op::OpImageSparseDrefGather] =
      DATA_DEP(1, {0, 2, 3, 4}, 6);
  mArgumentDependency[spv::Op::OpImageSparseTexelsResident] =
      DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpImageSparseRead] = DATA_DEP(1, {0, 2, 3}, 5);

  mArgumentDependency[spv::Op::OpCompositeConstruct] = DATA_DEP(1, {0}, 2);
  mArgumentDependency[spv::Op::OpCompositeExtract] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpCompositeInsert] = DATA_DEP(1, {0, 2, 3}, -1);
  mArgumentDependency[spv::Op::OpCopyObject] = DATA_DEP(1, {0, 2}, -1);

  mArgumentDependency[spv::Op::OpConvertUToF] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpConvertSToF] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpConvertFToU] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpConvertFToS] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpConvertPtrToU] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpConvertUToPtr] = DATA_DEP(1, {0, 2}, -1);

  mArgumentDependency[spv::Op::OpFDiv] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFMod] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFAdd] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFSub] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFMul] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFNegate] = DATA_DEP(1, {0}, -1);
  mArgumentDependency[spv::Op::OpFOrdEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFUnordEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFOrdNotEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFUnordNotEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFOrdLessThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFUnordLessThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFOrdGreaterThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFUnordGreaterThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFOrdLessThanEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFUnordLessThanEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFOrdGreaterThanEqual] =
      DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpFUnordGreaterThanEqual] =
      DATA_DEP(1, {0, 2}, -1);

  mArgumentDependency[spv::Op::OpIAdd] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpISub] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpIMul] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpSDiv] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpUDiv] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpSRem] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpINotEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpIEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpULessThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpSLessThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpUGreaterThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpSGreaterThan] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpULessThanEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpSLessThanEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpUGreaterThanEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpSGreaterThanEqual] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpShiftRightLogical] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpShiftRightArithmetic] =
      DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpShiftLeftLogical] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpBitwiseOr] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpBitwiseXor] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpBitwiseAnd] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpNot] = DATA_DEP(1, {0}, -1);

  mArgumentDependency[spv::Op::OpSampledImage] = DATA_DEP(1, {0, 2}, -1);
  mArgumentDependency[spv::Op::OpImage] = DATA_DEP(1, {0, 2}, -1);

  mArgumentDependency[spv::Op::OpVectorTimesScalar] =
      DATA_DEP(1, {0, 2, 3}, -1);
  mArgumentDependency[spv::Op::OpVectorTimesMatrix] =
      DATA_DEP(1, {0, 2, 3}, -1);
  mArgumentDependency[spv::Op::OpMatrixTimesVector] =
      DATA_DEP(1, {0, 2, 3}, -1);
  mArgumentDependency[spv::Op::OpMatrixTimesScalar] =
      DATA_DEP(1, {0, 2, 3}, -1);

  mArgumentDependency[spv::Op::OpSelect] = DATA_DEP(1, {0, 2, 3, 4}, -1);
}

void SpVcQuadGroupedIRGenerator::performConversionPassRegister() {
  mConvPassHandlers[spv::Op::OpLabel] = GET_CONV_PASS(OpLabel);
  mConvPassHandlers[spv::Op::OpFunction] = GET_CONV_PASS(OpFunction);
  mConvPassHandlers[spv::Op::OpBranch] = GET_CONV_PASS(OpBranch);
  mConvPassHandlers[spv::Op::OpBranchConditional] =
      GET_CONV_PASS(OpBranchConditional);
  mConvPassHandlers[spv::Op::OpSwitch] = GET_CONV_PASS(OpSwitch);

  mConvPassHandlers[spv::Op::OpFAdd] = GET_CONV_PASS(OpFAdd);
  mConvPassHandlers[spv::Op::OpFSub] = GET_CONV_PASS(OpFSub);
  mConvPassHandlers[spv::Op::OpFMul] = GET_CONV_PASS(OpFMul);
  mConvPassHandlers[spv::Op::OpFDiv] = GET_CONV_PASS(OpFDiv);

  mConvPassHandlers[spv::Op::OpIAdd] = GET_CONV_PASS(OpIAdd);
  mConvPassHandlers[spv::Op::OpISub] = GET_CONV_PASS(OpISub);
  mConvPassHandlers[spv::Op::OpIMul] = GET_CONV_PASS(OpIMul);

  mConvPassHandlers[spv::Op::OpFNegate] = GET_CONV_PASS(OpFNegate);
  mConvPassHandlers[spv::Op::OpFOrdLessThan] = GET_CONV_PASS(OpFOrdLessThan);
  mConvPassHandlers[spv::Op::OpFOrdGreaterThan] =
      GET_CONV_PASS(OpFOrdGreaterThan);
  mConvPassHandlers[spv::Op::OpFOrdLessThanEqual] =
      GET_CONV_PASS(OpFOrdLessThanEqual);
  mConvPassHandlers[spv::Op::OpFOrdGreaterThanEqual] =
      GET_CONV_PASS(OpFOrdGreaterThanEqual);
  mConvPassHandlers[spv::Op::OpFOrdEqual] = GET_CONV_PASS(OpFOrdEqual);
  mConvPassHandlers[spv::Op::OpFOrdNotEqual] = GET_CONV_PASS(OpFOrdNotEqual);
  mConvPassHandlers[spv::Op::OpFUnordLessThan] =
      GET_CONV_PASS(OpFUnordLessThan);
  mConvPassHandlers[spv::Op::OpFUnordGreaterThan] =
      GET_CONV_PASS(OpFUnordGreaterThan);
  mConvPassHandlers[spv::Op::OpFUnordLessThanEqual] =
      GET_CONV_PASS(OpFUnordLessThanEqual);
  mConvPassHandlers[spv::Op::OpFUnordGreaterThanEqual] =
      GET_CONV_PASS(OpFUnordGreaterThanEqual);
  mConvPassHandlers[spv::Op::OpFUnordEqual] = GET_CONV_PASS(OpFUnordEqual);
  mConvPassHandlers[spv::Op::OpFUnordNotEqual] =
      GET_CONV_PASS(OpFUnordNotEqual);

  mConvPassHandlers[spv::Op::OpSLessThan] = GET_CONV_PASS(OpSLessThan);
  mConvPassHandlers[spv::Op::OpSGreaterThan] = GET_CONV_PASS(OpSGreaterThan);
  mConvPassHandlers[spv::Op::OpSLessThanEqual] =
      GET_CONV_PASS(OpSLessThanEqual);
  mConvPassHandlers[spv::Op::OpSGreaterThanEqual] =
      GET_CONV_PASS(OpSGreaterThanEqual);
  mConvPassHandlers[spv::Op::OpULessThan] = GET_CONV_PASS(OpULessThan);
  mConvPassHandlers[spv::Op::OpUGreaterThan] = GET_CONV_PASS(OpUGreaterThan);
  mConvPassHandlers[spv::Op::OpULessThanEqual] =
      GET_CONV_PASS(OpULessThanEqual);
  mConvPassHandlers[spv::Op::OpUGreaterThanEqual] =
      GET_CONV_PASS(OpUGreaterThanEqual);
  mConvPassHandlers[spv::Op::OpIEqual] = GET_CONV_PASS(OpIEqual);
  mConvPassHandlers[spv::Op::OpINotEqual] = GET_CONV_PASS(OpINotEqual);

  mConvPassHandlers[spv::Op::OpVariable] = GET_CONV_PASS(OpVariable);
  mConvPassHandlers[spv::Op::OpLoad] = GET_CONV_PASS(OpLoad);
  mConvPassHandlers[spv::Op::OpStore] = GET_CONV_PASS(OpStore);

  mConvPassHandlers[spv::Op::OpCompositeConstruct] =
      GET_CONV_PASS(OpCompositeConstruct);
  mConvPassHandlers[spv::Op::OpCompositeExtract] =
      GET_CONV_PASS(OpCompositeExtract);
  mConvPassHandlers[spv::Op::OpExtInst] = GET_CONV_PASS(OpExtInst);

  // Stuffs to ignore
  mConvPassHandlers[spv::Op::OpDecorate] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpMemberDecorate] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeInt] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeFloat] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeVector] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeBool] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypePointer] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeVoid] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeFunction] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeSampler] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeSampledImage] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeImage] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeMatrix] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeArray] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpTypeStruct] = GET_CONV_PASS(DefOpIgnore);

  mConvPassHandlers[spv::Op::OpConstant] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpConstantComposite] = GET_CONV_PASS(DefOpIgnore);

  mConvPassHandlers[spv::Op::OpName] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpMemberName] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpSource] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpMemoryModel] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpExecutionMode] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpSelectionMerge] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpLoopMerge] = GET_CONV_PASS(DefOpIgnore);

  mConvPassHandlers[spv::Op::OpPhi] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpCapability] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpEntryPoint] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpExtInstImport] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpUnreachable] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpFunctionEnd] = GET_CONV_PASS(DefOpIgnore);
  mConvPassHandlers[spv::Op::OpSourceExtension] = GET_CONV_PASS(DefOpIgnore);

  mConvPassHandlers[spv::Op::OpDot] = GET_CONV_PASS(OpDot);
  mConvPassHandlers[spv::Op::OpVectorShuffle] = GET_CONV_PASS(OpVectorShuffle);

  mConvPassHandlers[spv::Op::OpAccessChain] = GET_CONV_PASS(OpAccessChain);
  mConvPassHandlers[spv::Op::OpReturn] = GET_CONV_PASS(OpReturn);

  mConvPassHandlers[spv::Op::OpVectorTimesScalar] =
      GET_CONV_PASS(OpVectorTimesScalar);

  mConvPassHandlers[spv::Op::OpDPdx] = GET_CONV_PASS(OpDPDx);
  mConvPassHandlers[spv::Op::OpDPdy] = GET_CONV_PASS(OpDPDy);
}

void SpVcQuadGroupedIRGenerator::performMaskInjectionPass(int quadSize) {
  setCurrentPass(SPVC_QGIR_MASKGEN);
}

void SpVcQuadGroupedIRGenerator::performTypeGenerationPass() {
#define VERBOSE_TPGEN(x) // EMIT_VERBOSE_CORE("Type generated: ",x->emitIR())
  for (auto k : mCtx->vO) {
    auto &v = mCtx->v[k];
    if (v.flag == SpVcBlockTypeVariableTypeFlag::SPVC_VARIABLE_TYPE) {
      if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_INT32) {
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeInt32>());
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_FLOAT32) {
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeFloat32>());
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_UNSIGNED32) {
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeInt32>());
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_BOOL) {
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeBool>());
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_POINTER) {
        auto ref = v.tp->children[0]->llvmType;
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypePointer>(ref));
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_VECTOR) {
        auto ref = v.tp->children[0]->llvmType;
        auto cnt = v.tp->size;
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeVector>(cnt, ref));
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_MATRIX) {
        auto ref = v.tp->children[0]->children[0]->llvmType;
        auto cnt = v.tp->size;
        auto childSize = v.tp->children[0]->size;
        auto ptr = addIr(
            std::make_unique<LLVM::SpVcLLVMTypeVector>(cnt * childSize, ref));
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_ARRAY) {
        auto ref = v.tp->children[0]->llvmType;
        auto cnt = v.tp->size;
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeArray>(cnt, ref));
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_STRUCT) {
        std::vector<LLVM::SpVcLLVMType *> ch{};
        for (int i = 0; i < v.tp->children.size(); i++) {
          ch.push_back(v.tp->children[i]->llvmType);
        }
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeStruct>(ch));
        auto aliasName = allocateLlvmVarName();
        auto aliasNameExpr =
            addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(aliasName));
        auto typedIns = addIrG(
            std::make_unique<LLVM::SpVcLLVMIns_TypeAlias>(aliasNameExpr, ptr));
        v.tp->llvmType =
            addIr(std::make_unique<LLVM::SpVcLLVMTypeStructAliased>(aliasName));
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_IMAGE) {
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeVoidPtr>());
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_SAMPLER) {
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeVoidPtr>());
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else if (v.tp->type == SpVcVMTypeEnum::SPVC_TYPE_SAMPLED_IMAGE) {
        auto ptr = addIr(std::make_unique<LLVM::SpVcLLVMTypeVoidPtr>());
        v.tp->llvmType = ptr;
        VERBOSE_TPGEN(ptr);
      } else {
        EMIT_WARN_CORE("Unknown type encountered:", (int)v.tp->type,
                       ", At variable %", v.id);
      }
    }
  }

  // Then for all variables, allocate 4 elements
  for (auto k : mCtx->vO) {
    auto &v = mCtx->v[k];
    if (v.flag & (SPVC_VARIABLE_VAR | SPVC_VARIABLE_TEMP)) {
      auto tp = v.tpRef->tp->llvmType;
      v.llvmVarName.resize(getQuads());
      for (int i = 0; i < getQuads(); i++) {
        if (v.blockBelong == nullptr) {
          auto varName =
              addIr(std::make_unique<LLVM::SpVcLLVMGlobalVariableName>(
                  allocateLlvmVarName()));
          auto varArg =
              addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, varName));
          v.llvmVarName[i].arg = varArg;
          // EMIT_VERBOSE_CORE("Allocated llvmir for", k, " : ",
          // varArg->emitIR());
        } else {
          auto varName =
              addIr(std::make_unique<LLVM::SpVcLLVMLocalVariableName>(
                  allocateLlvmVarName()));
          auto varArg =
              addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tp, varName));
          v.llvmVarName[i].arg = varArg;
          // EMIT_VERBOSE_CORE("Allocated llvmir for", k, " : ",
          // varArg->emitIR());
        }
      }
      if (v.descSet != nullptr) {
        if (v.storageClass == spv::StorageClass::StorageClassInput) {
          if (v.descSet->location == -1) {
            EMIT_ERROR_CORE("Input variable has no location specified");
          }
          int loc = v.descSet->location;
          for (int i = 0; i < getQuads(); i++) {
            mCtx->binds.unordInputVars[i][loc] =
                IrUtil::symbolClean(v.llvmVarName[i].arg->emitIRName());
            mCtx->binds.unordInputVarsSz[i][loc] =
                IrUtil::getTypeSize(v.tpRef->tp.get());
          }
        } else if (v.storageClass == spv::StorageClass::StorageClassOutput) {
          if (v.descSet->location == -1) {
            EMIT_ERROR_CORE("Output variable has no location specified");
          }
          int loc = v.descSet->location;
          for (int i = 0; i < getQuads(); i++) {
            mCtx->binds.unordOutputVars[i][loc] =
                IrUtil::symbolClean(v.llvmVarName[i].arg->emitIRName());
            mCtx->binds.unordOutputVarsSz[i][loc] =
                IrUtil::getTypeSize(v.tpRef->tp.get());
          }
        } else if (v.storageClass == spv::StorageClass::StorageClassUniform) {
          if (v.descSet->descriptorSet == -1 || v.descSet->binding == -1) {
            EMIT_ERROR_CORE("Uniform variable has no location specified");
          }
          mCtx->binds.uniformVarSymbols.push_back(
              IrUtil::symbolClean(v.llvmVarName[0].arg->emitIRName()));
          mCtx->binds.uniformVarLoc.push_back(
              {v.descSet->descriptorSet, v.descSet->binding});
          mCtx->binds.uniformVarSz.push_back(
              IrUtil::getTypeSize(v.tpRef->tp.get()));
        }
      }
    }
    if (v.flag & (SPVC_VARIABLE_CONSTANT)) {
      auto tp = v.tpRef->tp.get();
      auto tpllvm = v.tpRef->tp->llvmType;
      if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_VECTOR) {
        std::vector<LLVM::SpVcLLVMArgument *> args;
        for (int i = 0; i < v.constant->children.size(); i++) {
          args.push_back(v.constant->children[i]->arg);
        }
        auto constv =
            addIr(std::make_unique<LLVM::SpVcLLVMConstantValueVector>(args));
        auto constArg =
            addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpllvm, constv));
        v.constant->arg = constArg;
      } else if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_INT32) {
        auto constv = addIr(std::make_unique<LLVM::SpVcLLVMConstantValueInt>(
            v.constant->value[0]));
        auto constArg =
            addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpllvm, constv));
        v.constant->arg = constArg;
      } else if (tp->type == SpVcVMTypeEnum::SPVC_TYPE_FLOAT32) {
        auto constv =
            addIr(std::make_unique<LLVM::SpVcLLVMConstantValueFloat32>(
                v.constant->value[0]));
        auto constArg =
            addIr(std::make_unique<LLVM::SpVcLLVMArgument>(tpllvm, constv));
        v.constant->arg = constArg;
      }
    }
  }

  // For each block, an entry label is required `
  for (auto &block : mCtx->blocks) {
    auto varName =
        addIrB(std::make_unique<LLVM::SpVcLLVMLabelName>(allocateLlvmVarName()),
               block.get());
    block->llvmLabel = varName;
    // EMIT_VERBOSE_CORE("Allocated llvmir for block entry label: ",
    // varName->emitIR());
  }

  // For all phi instructions, register dependencies
  for (int i = 0; i < mRaw->instructions.size(); i++) {
    auto &ins = mRaw->instructions[i];
    if (ins.opCode == spv::Op::OpPhi) {
      for (int j = 2; j < ins.opParams.size(); j++) {
        auto var = getVariableSafe(ins.opParams[j]);
        if (var->flag & (SPVC_VARIABLE_VAR | SPVC_VARIABLE_TEMP)) {
          if (var->blockBelong !=
              getVariableSafe(ins.opParams[j + 1])->blockBelong) {
            getVariableSafe(ins.opParams[j + 1])
                ->phiDepsEx.push_back({var, getVariableSafe(ins.opParams[1])});
          } else {
            var->phiDeps.push_back(getVariableSafe(ins.opParams[1]));
          }
        } else if (var->flag & (SPVC_VARIABLE_CONSTANT)) {
          getVariableSafe(ins.opParams[j + 1])
              ->phiDepsEx.push_back({var, getVariableSafe(ins.opParams[1])});
        }
      }
    }
  }
}

void SpVcQuadGroupedIRGenerator::performBlockPass() {
  setCurrentPass(SPVC_QGIR_BLOCK_GENERATION);
  for (int i = 0; i < mRaw->instructions.size(); i++) {
    auto &inst = mRaw->instructions[i];
    setCurrentProgCounter(i);
    if (inst.opCode == spv::Op::OpLabel) {
      mCtx->blocks.push_back(std::make_unique<SpVcVMGenBlock>());
      auto &block = mCtx->blocks.back();
      block->startingPc = i;
      block->funcBelong = mCtx->activeFuncEnv;
      if (mCtx->activeFuncEnv != nullptr) {
        mCtx->activeFuncEnv->blocks.push_back(block.get());
      }
      getVariableSafe(inst.opParams[0])->blockBelong =
          mCtx->blocks.back().get();
    }
    if (inst.opCode == spv::Op::OpFunction) {
      mCtx->funcs.push_back(std::make_unique<SpVcVMGenFunction>());
      auto &func = mCtx->funcs.back();
      func->startingPc = i;
      mCtx->activeFuncEnv = func.get();
      getVariableSafe(inst.opParams[1])->funcBelong = func.get();
    }
    if (inst.opCode == spv::Op::OpFunctionEnd) {
      mCtx->activeFuncEnv = nullptr;
    }
  }
  mCtx->blockStack.clear();
  mCtx->activeFuncEnv = nullptr;
}

void SpVcQuadGroupedIRGenerator::init() {
  performDefinitionPassRegister();
  performDataflowResolutionPassRegister();
  performConversionPassRegister();
}

void SpVcQuadGroupedIRGenerator::verbose() {
  printf("====== Blocks ====== \n");
  for (int i = 0; i < mCtx->blocks.size(); i++) {
    printf("[%d] Block %d\n", i, mCtx->blocks[i]->startingPc);
    printf("  Predecessors: ");
    for (int j = 0; j < mCtx->blocks[i]->cfgPredecessor.size(); j++) {
      printf("%d ", mCtx->blocks[i]->cfgPredecessor[j]->startingPc);
    }
    printf("\n");
    printf("  Successors: ");
    for (int j = 0; j < mCtx->blocks[i]->cfgSuccessor.size(); j++) {
      printf("%d ", mCtx->blocks[i]->cfgSuccessor[j]->startingPc);
    }
    printf("\n");
    printf("  Block Tags: ");
    for (int j = 0; j < mCtx->blocks[i]->blockType.size(); j++) {
      switch (mCtx->blocks[i]->blockType[j].blockType) {
      case SPVC_BLOCK_SELECTION_HEADER:
        printf("<Selection Header %d> ",
               mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_SELECTION_MERGE:
        printf("<Selection Merge %d> ",
               mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_SELECTION_BODY_FIRST:
        printf("<Selection Body <True> %d> ",
               mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_SELECTION_BODY_SECOND:
        printf("<Selection Body <False> %d> ",
               mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_SELECTION_BODY_SWITCH:
        printf("<Selection Body <Switch> %d> ",
               mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_LOOP_HEADER:
        printf("<Loop Header %d> ", mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_LOOP_MERGE:
        printf("<Loop Merge %d> ", mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_LOOP_BODY:
        printf("<Loop Body %d> ", mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_LOOP_CONTINUE:
        printf("<Loop Continue %d> ",
               mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_LOOP_BREAK:
        printf("<Loop Break %d> ", mCtx->blocks[i]->blockType[j].progCounter);
        break;
      case SPVC_BLOCK_RETURN:
        printf("<Return %d> ", mCtx->blocks[i]->blockType[j].progCounter);
        break;
      }
    }
    printf("\n");
  }
  printf("====== Registers ====== \n");
  for (auto &[k, v] : mCtx->v) {
    printf("[%d] In Block %d\n", v.id,
           (v.blockBelong) ? v.blockBelong->startingPc : -1);
    printf("  Rely On: ");
    for (auto &v2 : v.dependOnVars) {
      printf("%d ", v2);
    }
    printf("\n");
    printf("  Used By: ");
    for (auto &v2 : v.usedByVars) {
      printf("%d ", v2);
    }
    printf("\n");
  }
  printf("====== Gen IR ====== \n");
  printf("%s\n", generateIR().c_str());

  printf("====== Shader Inputs/Outputs ====== \n");
  for (int i = 0; i < getQuads(); i++) {
    printf("Quad %d\n", i);
    printf("  Inputs: ");
    for (auto &[k, v] : mCtx->binds.unordInputVars[i]) {
      printf("%d:%s ", k, v.c_str());
    }
    printf("\n");
    printf("  Outputs: ");
    for (auto &[k, v] : mCtx->binds.unordOutputVars[i]) {
      printf("%d:%s ", k, v.c_str());
    }
    printf("\n");
  }
  // Print Size
  for (int i = 0; i < getQuads(); i++) {
    printf("Quad %d\n", i);
    printf("  Inputs Sizes:");
    for (auto &[k, v] : mCtx->binds.unordInputVarsSz[i]) {
      printf("%d:%d ", k, v);
    }
    printf("\n");
    printf("  Outputs Sizes:");
    for (auto &[k, v] : mCtx->binds.unordOutputVarsSz[i]) {
      printf("%d:%d ", k, v);
    }
    printf("\n");
  }

  printf("Uniforms: ");
  for (int i = 0; i < mCtx->binds.uniformVarSymbols.size(); i++) {
    printf("%s(<%d,%d>,%d)", mCtx->binds.uniformVarSymbols[i].c_str(),
           mCtx->binds.uniformVarLoc[i].first,
           mCtx->binds.uniformVarLoc[i].second, mCtx->binds.uniformVarSz[i]);
  }
}

void SpVcQuadGroupedIRGenerator::performDefinitionPass() {
  setCurrentPass(SPVC_QGIR_DEFINITION);
  auto &ins = mRaw->instructions;
  for (int i = 0; i < ins.size(); i++) {
    setCurrentProgCounter(i);
    bool hasRtype = false, hasRes = false;
    spv::HasResultAndType((spv::Op)ins[i].opCode, &hasRes, &hasRtype);
    if (hasRes && !hasRtype) {
      mCtx->vO.push_back(ins[i].opParams[0]);
    }
    if (hasRes && hasRtype) {
      mCtx->vO.push_back(ins[i].opParams[1]);
    }
    if (mDefinitionPassHandlers.count(ins[i].opCode) > 0) {
      mDefinitionPassHandlers[ins[i].opCode](i, ins[i].opParams, mCtx, this);
    } else if (hasRes && hasRtype) {
      mUniversalDefinitionPassHandler(i, ins[i].opParams, mCtx, this);
    } else {
      EMIT_ERROR_CORE("Unknown opcode: ", ins[i].opCode,
                      spv::OpToString((spv::Op)ins[i].opCode));
    }
  }
}

void SpVcQuadGroupedIRGenerator::performConversionPass() {
  setCurrentPass(SPVC_QGIR_CONVERSION);
  mCtx->blockStack.clear();
  auto &ins = mRaw->instructions;
  for (int i = 0; i < ins.size(); i++) {
    setCurrentProgCounter(i);
    // ADD NOTE
    if (getActiveBlock() != nullptr) {
      addIrB(
          std::make_unique<LLVM::SpVcLLVMIns_Note>("line " + std::to_string(i)),
          getActiveBlock());
    }
    // EMIT
    if (mConvPassHandlers.count(ins[i].opCode) > 0) {
      bool hasRtype = false, hasRes = false;
      spv::HasResultAndType((spv::Op)ins[i].opCode, &hasRes, &hasRtype);
      if (hasRes && hasRtype && getActiveBlock()) {
        // Add alloc instuctions first
        for (int k = 0; k < getQuads(); k++) {
          auto var = getVariableSafe(ins[i].opParams[1]);
          auto varName = var->llvmVarName[k].arg;
          auto varType = var->tpRef->tp->llvmType;
          addIrB(std::make_unique<LLVM::SpVcLLVMIns_Alloca>(varName, varType),
                 getActiveBlock());
        }
      }

      mConvPassHandlers[ins[i].opCode](i, ins[i].opParams, mCtx, this);

      if (hasRes && hasRtype) {
        auto var = getVariableSafe(ins[i].opParams[1]);
        for (auto &phiDep : var->phiDeps) {
          addIrB(std::make_unique<LLVM::SpVcLLVMIns_Note>("Phi Dep"),
                 getActiveBlock());
          for (int k = 0; k < getQuads(); k++) {
            auto immReg =
                ConversionPass::spirvImmediateLoad(ins[i].opParams[1], k, this);
            ConversionPass::spirvImmediateMaskedStore(phiDep->id, k, immReg,
                                                      this);
          }
        }
      }
      if (ins[i].opCode == spv::Op::OpLabel) {
        auto var = getVariableSafe(ins[i].opParams[0]);
        for (auto &phiDep : var->phiDepsEx) {
          addIrB(std::make_unique<LLVM::SpVcLLVMIns_Note>("Phi Dep"),
                 getActiveBlock());
          for (int k = 0; k < getQuads(); k++) {
            auto immReg =
                ConversionPass::spirvImmediateLoad(phiDep.first->id, k, this);
            ConversionPass::spirvImmediateMaskedStore(phiDep.second->id, k,
                                                      immReg, this);
          }
        }
      }

    } else {
      EMIT_WARN_CORE("Unknown opcode: ", ins[i].opCode,
                     spv::OpToString((spv::Op)ins[i].opCode));
    }
  }
}

void SpVcQuadGroupedIRGenerator::performDataflowResolutionPass() {
  setCurrentPass(SPVC_QGIR_DATAFLOW_DEPENDENCY);
  auto &ins = mRaw->instructions;
  for (int i = 0; i < ins.size(); i++) {
    setCurrentProgCounter(i);
    if (ins[i].opCode == spv::OpLabel) {
      auto block = GET_PARAM_CORE(ins[i].opParams[0])->blockBelong;
      pushActiveBlock(block);
    }

    if (mArgumentDependency.count(ins[i].opCode) > 0) {
      if (!mArgumentDependency.count(ins[i].opCode)) {
        EMIT_ERROR_CORE("Unknown opcode: ", ins[i].opCode);
      }
      auto &dep = mArgumentDependency[ins[i].opCode];
      if (dep.special == SPVC_DATADEPS_ORDINARY) {
        SpVcVMGenVariable *dataOutput = nullptr;
        if (dep.retArg != -1) {
          dataOutput = GET_PARAM_CORE(ins[i].opParams[dep.retArg]);
        } else {
          // EMIT_VERBOSE_CORE("No return value for ", ins[i].opCode, " at ",
          // i);
        }
        for (int j = 0; j < dep.depArgs.size(); j++) {
          int itx = dep.depArgs[j];
          if (itx < ins[i].opParams.size()) {
            auto dataDependOn = GET_PARAM_CORE(ins[i].opParams[itx]);
            if (dataOutput != nullptr) {
              dataOutput->dependOnVars.insert(ins[i].opParams[itx]);
              dataDependOn->usedByVars.insert(ins[i].opParams[dep.retArg]);
            }
            if (getActiveBlock()) {
              getActiveBlock()->dependOnVar.insert(ins[i].opParams[itx]);
            }
          }
        }
        if (dep.depVaArgs != -1) {
          for (int itx = dep.depVaArgs; itx < ins[i].opParams.size(); itx++) {
            auto dataDependOn = GET_PARAM_CORE(ins[i].opParams[itx]);
            if (dataOutput != nullptr) {
              dataOutput->dependOnVars.insert(ins[i].opParams[itx]);
              dataDependOn->usedByVars.insert(ins[i].opParams[dep.retArg]);
            }
            if (getActiveBlock()) {
              getActiveBlock()->dependOnVar.insert(ins[i].opParams[itx]);
            }
          }
        }
      } else if (dep.special == SPVC_DATADEPS_SSAPHI) {
        // SSA PHI
        auto dataOutput = GET_PARAM_CORE(ins[i].opParams[1]);
        for (int j = 2; j < ins[i].opParams.size(); j += 2) {
          int itx = dep.depArgs[j];
          if (itx < ins[i].opParams.size()) {
            auto dataDependOn = GET_PARAM_CORE(ins[i].opParams[j]);
            dataOutput->dependOnVars.insert(ins[i].opParams[j]);
            dataDependOn->usedByVars.insert(ins[i].opParams[0]);
          }
          if (getActiveBlock()) {
            getActiveBlock()->dependOnVar.insert(ins[i].opParams[0]);
          }
        }
      }
    }
  }
}

void SpVcQuadGroupedIRGenerator::performSymbolExport() {
  // Shader inputs
  for (int i = 0; i < getQuads(); i++) {
    for (int j = 0; j < mCtx->binds.unordInputVars[i].size(); j++) {
      if (mCtx->binds.unordInputVars[i].count(j) == 0) {
        EMIT_ERROR_CORE("Input variable not found: ", j);
      }
      auto varName = mCtx->binds.unordInputVars[i][j];
      auto varSz = mCtx->binds.unordInputVarsSz[i][j];
      mCtx->binds.inputVarSymbols[i].push_back(varName);
      mCtx->binds.inputSize[i].push_back(varSz);
    }
  }
  // Shader outputs
  for (int i = 0; i < getQuads(); i++) {
    for (int j = 0; j < mCtx->binds.unordOutputVars[i].size(); j++) {
      if (mCtx->binds.unordOutputVars[i].count(j) == 0) {
        EMIT_ERROR_CORE("Output variable not found: ", j);
      }
      auto varName = mCtx->binds.unordOutputVars[i][j];
      auto varSz = mCtx->binds.unordOutputVarsSz[i][j];
      mCtx->binds.outputVarSymbols[i].push_back(varName);
      mCtx->binds.outputSize[i].push_back(varSz);
    }
  }
}

void SpVcQuadGroupedIRGenerator::parse() {
  performBlockPass();
  performDefinitionPass();
  performDataflowResolutionPass();
  performTypeGenerationPass();
  performConversionPass();
  performSymbolExport();
  ifritLog1("IR Generation Complete");
}

std::string SpVcQuadGroupedIRGenerator::generateIR() {
  std::string ret = "";
  ret += "declare float @llvm.sin.f32(float %Val)\n";
  ret += "declare float @llvm.cos.f32(float %Val)\n";
  ret += "declare float @llvm.tan.f32(float %Val)\n";
  ret += "declare float @llvm.asin.f32(float %Val)\n";
  ret += "declare float @llvm.acos.f32(float %Val)\n";
  ret += "declare float @llvm.atan.f32(float %Val)\n";
  ret += "declare float @llvm.sqrt.f32(float %Val)\n";
  ret += "declare float @llvm.fma.f32(float %Val,float %Val2,float %Val3)\n";
  ret += "declare float @llvm.fma.v4f32( <4 x float> %Val, <4 x float> %Val2, "
         "<4 x float> %Val3)\n";
  ret += "attributes #0 = { noinline nounwind \"no-stack-arg-probe\" }\n";
  ret += mExtInstGen.getRequiredFuncDefs() + "\n";

  ret += "; global section\n";
  for (auto &x : mCtx->globalDefs) {
    ret += x->emitIR() + "\n";
  }
  for (auto &f : mCtx->funcs) {
    ret += "; function " + std::to_string(f->startingPc) + "\n";
    for (auto &x : f->ir) {
      ret += x->emitIR() + "\n";
    }
    for (auto &b : f->blocks) {
      ret += "; block " + std::to_string(b->startingPc) + "\n";
      for (auto &x : b->irPre) {
        ret += x->emitIR() + "\n";
      }
      for (auto &x : b->ir) {
        ret += x->emitIR() + "\n";
      }
    }
    for (auto &x : f->irPost) {
      ret += x->emitIR() + "\n";
    }
  }

  return ret;
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec