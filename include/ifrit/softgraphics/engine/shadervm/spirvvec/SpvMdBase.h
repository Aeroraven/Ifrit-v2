#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMContext.h"
#include "ifrit/softgraphics/engine/shadervm/spirvvec/SpvMdLlvmIrRepr.h"
#include <stack>

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec {
constexpr int SpVcQuadSize = 4;
enum class SpVcVMTypeEnum {
  SPVC_TYPE_UNDEFINED,
  SPVC_TYPE_BOOL,
  SPVC_TYPE_VOID,
  SPVC_TYPE_INT32,
  SPVC_TYPE_FLOAT32,
  SPVC_TYPE_UNSIGNED32,
  SPVC_TYPE_FLOAT64,
  SPVC_TYPE_INT64,
  SPVC_TYPE_UNSIGNED64,
  SPVC_TYPE_INT8,
  SPVC_TYPE_UNSIGNED8,
  SPVC_TYPE_FLOAT8,
  SPVC_TYPE_INT16,
  SPVC_TYPE_UNSIGNED16,
  SPVC_TYPE_FLOAT16,
  SPVC_TYPE_VECTOR,
  SPVC_TYPE_MATRIX,
  SPVC_TYPE_STRUCT,
  SPVC_TYPE_ARRAY,
  SPVC_TYPE_POINTER,
  SPVC_TYPE_FUNCTION,
  SPVC_TYPE_IMAGE,
  SPVC_TYPE_SAMPLER,
  SPVC_TYPE_SAMPLED_IMAGE
};

// Specifies structural block types
enum SpVcBlockTypeEnum {
  SPVC_BLOCK_UNDEFINED = 0x0,
  SPVC_BLOCK_SELECTION_HEADER = 0x1,
  SPVC_BLOCK_SELECTION_BODY_FIRST = 0x2,
  SPVC_BLOCK_SELECTION_MERGE = 0x4,
  SPVC_BLOCK_LOOP_HEADER = 0x8,
  SPVC_BLOCK_LOOP_BREAK =
      0x10, // Loop break contains branch to loop merge section
  SPVC_BLOCK_LOOP_CONTINUE = 0x20,
  SPVC_BLOCK_LOOP_BODY = 0x40,
  SPVC_BLOCK_LOOP_MERGE = 0x80,
  SPVC_BLOCK_RETURN = 0x100,
  SPVC_BLOCK_SELECTION_BODY_SECOND = 0x200,
  SPVC_BLOCK_SELECTION_BODY_SWITCH = 0x400,
};

enum SpVcBlockTypeVariableTypeFlag {
  SPVC_VARIABLE_TEMP = 0x1,    // Temporary, in registers
  SPVC_VARIABLE_INPUT = 0x2,   // Input, from external
  SPVC_VARIABLE_OUTPUT = 0x4,  // Output, to external
  SPVC_VARIABLE_UNIFORM = 0x8, // Uniform, from external
  SPVC_VARIABLE_CONSTANT = 0x10,
  SPVC_VARIABLE_TYPE = 0x20,
  SPVC_VARIABLE_EXTINST_IMPORT = 0x40,
  SPVC_VARIABLE_VAR = 0x80,
  SPVC_VARIABLE_ACCESS_CHAIN = 0x100,
};

enum SpVcStructuredControlFlowIndication {
  SPVC_STRUCTCFG_NONE = 0x0,
  SPVC_STRUCTCFG_SELECTION_MERGE = 0x1,
  SPVC_STRUCTCFG_LOOP_MERGE = 0x2,
};

constexpr int SPVC_QUAD_SIZE = 4;

class SpVcGenInstruction;
struct SpVcVMGenBlock;
struct SpVcVMGenVariable;
struct SpVcVMGenFunction;
struct SpVcVMGenStack;

struct SpVcVMTypeDescriptor {
  std::vector<SpVcVMTypeDescriptor *> children;
  std::string name;
  int size;
  SpVcVMTypeEnum type = SpVcVMTypeEnum::SPVC_TYPE_UNDEFINED;
  int storageClass;
  LLVM::SpVcLLVMType *llvmType;
  bool isUnsigned = false;
};

// Holds compile-time constant values
struct SpVcVMGenConstant {
  SpVcVMTypeDescriptor *tpRef;
  std::vector<SpVcVMGenConstant *> children;
  std::vector<int> value;
  LLVM::SpVcLLVMArgument *arg;
};

struct SpVcVMDecoration {
  int binding = -1;
  int location = -1;
  int descriptorSet = -1;
};

struct SpVcGenIRVariable {
  std::string name;
  LLVM::SpVcLLVMArgument *arg;
  LLVM::SpVcLLVMArgument *finalReg;
};

// Holds SSA Variables
struct SpVcVMGenVariable {
  SpVcVMGenVariable *tpRef = nullptr;
  std::string name;
  SpVcGenInstruction *def;
  SpVcVMGenBlock *blockBelong = nullptr;
  SpVcVMGenFunction *funcBelong = nullptr;
  std::unique_ptr<SpVcVMTypeDescriptor> tp;
  std::unique_ptr<SpVcVMGenConstant> constant;
  std::unique_ptr<SpVcVMDecoration> descSet;
  std::unordered_set<int> usedByVars;
  std::unordered_set<int> dependOnVars;
  int flag = 0;
  int id = -1;
  bool isAdditional = false;
  int storageClass;

  bool isAllQuad = false;
  std::vector<SpVcGenIRVariable> llvmVarName;
  std::vector<SpVcVMGenVariable *> phiDeps;
  std::vector<std::pair<SpVcVMGenVariable *, SpVcVMGenVariable *>> phiDepsEx;
};

// Execution mask guarantees the SIMT-like synchronization
struct SpVcVMExecutionMask {
  SpVcVMGenVariable *continueMask = nullptr;
  SpVcVMGenVariable *breakMask = nullptr;
  SpVcVMGenVariable *returnMask = nullptr;
  SpVcVMGenVariable *execMask = nullptr;

  std::vector<SpVcVMGenVariable *> activeExecMask;
  std::vector<SpVcVMGenVariable *> branchMask;
};

struct SpVcBlockTypeRecord {
  int blockType;
  int progCounter;
  bool operator==(const SpVcBlockTypeRecord &other) const {
    return blockType == other.blockType && progCounter == other.progCounter;
  }
};

// Block records variables that might be used by other blocks (for masking
// considerations) and record dependencies of blocks
struct SpVcVMGenBlock {
  std::vector<SpVcBlockTypeRecord> blockType;
  std::string label;
  std::vector<SpVcVMGenVariable *> variables;
  std::vector<SpVcVMGenVariable *> exportingVariables;
  std::vector<SpVcGenInstruction *> instructions;
  bool isMainFunction;
  int startingPc;

  std::vector<SpVcVMGenBlock *> cfgSuccessor;
  std::vector<SpVcVMGenBlock *> cfgPredecessor;
  std::unordered_set<int> dependOnVar;
  SpVcVMGenFunction *funcBelong = nullptr;

  LLVM::SpVcLLVMLabelName *llvmLabel;
  std::vector<LLVM::SpVcLLVMExpr *> ir;
  std::vector<LLVM::SpVcLLVMExpr *> irPre;

  LLVM::SpVcLLVMLabelName *contLabel;
  LLVM::SpVcLLVMLabelName *loopStartLabel;
  SpVcVMGenStack *stackBelong = nullptr;
};

struct SpVcVMGenFunction {
  int startingPc;
  bool isMainFunction;
  std::vector<SpVcVMGenBlock *> blocks;
  std::vector<LLVM::SpVcLLVMExpr *> ir;
  std::vector<LLVM::SpVcLLVMExpr *> irPost;
  SpVcVMGenVariable *returnMask = nullptr;
};

struct SpVcVMGenStack {
  SpVcVMGenBlock *header;
  SpVcVMExecutionMask masks;
  int ifStackSize = 0;
};

// Entrypoint
struct SpVcVMEntryPoint {
  int execModel;
  int entryPoint;
  std::string name;
  std::vector<int> input;
};

// Memory model
struct SpVcVMMemoryModel {
  int addressingModel;
  int memoryModel;
};

// Exporting Symbols
struct SpVcSymbolInfo {
  std::string mainFunction;
  std::vector<std::string> inputVarSymbols[SpVcQuadSize];
  std::vector<std::string> outputVarSymbols[SpVcQuadSize];
  std::vector<std::string> uniformVarSymbols;
  std::vector<std::pair<int, int>> uniformVarLoc;

  std::vector<int> inputSize[SpVcQuadSize];
  std::vector<int> outputSize[SpVcQuadSize];
  std::vector<int> uniformVarSz;

  std::map<int, std::string> unordInputVars[SpVcQuadSize];
  std::map<int, std::string> unordOutputVars[SpVcQuadSize];
  std::map<int, int> unordInputVarsSz[SpVcQuadSize];
  std::map<int, int> unordOutputVarsSz[SpVcQuadSize];
};

// Generator context
struct SpVcVMGeneratorContext {
  std::vector<int> vO;
  std::unordered_map<int, SpVcVMGenVariable> v;
  std::vector<std::unique_ptr<SpVcVMGenBlock>> blocks;
  std::vector<std::unique_ptr<SpVcVMGenStack>> genstack;
  std::vector<std::unique_ptr<SpVcVMGenFunction>> funcs;
  SpVcStructuredControlFlowIndication cfgInd;

  std::vector<SpVcVMEntryPoint> entryPoints;
  SpVcVMMemoryModel memoryModel;
  std::vector<int> capabilities;

  SpVcVMGenFunction *activeFuncEnv;
  std::vector<SpVcVMGenBlock *> blockStack;

  std::vector<SpVcVMGenStack *> structStack;

  SpVcVMGenVariable *maskTypeRef;
  std::vector<std::unique_ptr<LLVM::SpVcLLVMExpr>> irExprs;
  std::vector<LLVM::SpVcLLVMExpr *> globalDefs;

  int funcCounter = 0;
  SpVcSymbolInfo binds;
};

} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec