#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMContext.h"
#include <tuple>

namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec::LLVM {
class SpVcLLVMExpr {
public:
  virtual ~SpVcLLVMExpr() = default;
  virtual std::string emitIR() = 0;
};

template <class... Tp> class SpVcLLVMCompContainer {
protected:
  std::tuple<Tp...> mComponents;

public:
  virtual ~SpVcLLVMCompContainer() = default;
  SpVcLLVMCompContainer(Tp... components) : mComponents(components...) {}
};

// Base Types
class SpVcLLVMType : public virtual SpVcLLVMExpr {
public:
  virtual ~SpVcLLVMType() = default;
  virtual std::string emitIR() override = 0;
};
class SpVcLLVMBaseType : public virtual SpVcLLVMType {
public:
  virtual ~SpVcLLVMBaseType() = default;
  virtual std::string emitIR() override = 0;
};
class SpVcLLVMTypeInt32 : public virtual SpVcLLVMBaseType {
public:
  virtual ~SpVcLLVMTypeInt32() = default;
  std::string emitIR() override { return "i32"; }
};
class SpVcLLVMTypeFloat32 : public SpVcLLVMBaseType {
public:
  virtual ~SpVcLLVMTypeFloat32() = default;
  std::string emitIR() override { return "float"; }
};
class SpVcLLVMTypeBool : public SpVcLLVMBaseType {
public:
  virtual ~SpVcLLVMTypeBool() = default;
  std::string emitIR() override { return "i1"; }
};
class SpVcLLVMTypeVoid : public SpVcLLVMBaseType {
public:
  virtual ~SpVcLLVMTypeVoid() = default;
  std::string emitIR() override { return "void"; }
};
class SpVcLLVMTypeVoidPtr : public SpVcLLVMBaseType {
public:
  virtual ~SpVcLLVMTypeVoidPtr() = default;
  std::string emitIR() override { return "i8*"; }
};
class SpVcLLVMTypePointer : public SpVcLLVMType {
  SpVcLLVMType *mBaseType;

public:
  virtual ~SpVcLLVMTypePointer() = default;
  SpVcLLVMTypePointer(SpVcLLVMType *baseType) : mBaseType(baseType) {}
  std::string emitIR() override { return mBaseType->emitIR() + "*"; }
};
class SpVcLLVMTypeVector : public SpVcLLVMType,
                           public SpVcLLVMCompContainer<int, SpVcLLVMType *> {
public:
  virtual ~SpVcLLVMTypeVector() = default;
  SpVcLLVMTypeVector(int size, SpVcLLVMType *baseType)
      : SpVcLLVMCompContainer(size, baseType) {}
  std::string emitIR() override {
    return "<" + std::to_string(std::get<0>(mComponents)) + " x " +
           std::get<1>(mComponents)->emitIR() + ">";
  }
};

class SpVcLLVMTypeArray : public SpVcLLVMType,
                          public SpVcLLVMCompContainer<int, SpVcLLVMType *> {
public:
  virtual ~SpVcLLVMTypeArray() = default;
  SpVcLLVMTypeArray(int size, SpVcLLVMType *baseType)
      : SpVcLLVMCompContainer(size, baseType) {}
  std::string emitIR() override {
    return "[" + std::to_string(std::get<0>(mComponents)) + " x " +
           std::get<1>(mComponents)->emitIR() + "]";
  }
};

class SpVcLLVMTypeStruct : public SpVcLLVMType {
  std::vector<SpVcLLVMType *> mMembers;

public:
  virtual ~SpVcLLVMTypeStruct() = default;
  SpVcLLVMTypeStruct(std::vector<SpVcLLVMType *> members) : mMembers(members) {}
  std::string emitIR() override {
    std::string ret = "{";
    for (auto &member : mMembers) {
      ret += member->emitIR() + ", ";
    }
    ret.pop_back();
    ret.pop_back();
    ret += "}";
    return ret;
  }
};

class SpVcLLVMTypeStructAliased : public SpVcLLVMType {
  std::string mName;

public:
  virtual ~SpVcLLVMTypeStructAliased() = default;
  SpVcLLVMTypeStructAliased(std::string name) : mName(name) {}
  std::string emitIR() override { return "%" + mName; }
};

// Local variable
class SpVcLLVMVarName : public SpVcLLVMExpr {
public:
  virtual ~SpVcLLVMVarName() = default;
  virtual std::string emitIR() override = 0;
};

class SpVcLLVMLocalVariableName : public SpVcLLVMVarName {
  std::string mName;

public:
  virtual ~SpVcLLVMLocalVariableName() = default;
  SpVcLLVMLocalVariableName(std::string name) : mName(name) {}
  std::string emitIR() override { return "%" + mName; }
};

class SpVcLLVMGlobalVariableName : public SpVcLLVMVarName {
  std::string mName;

public:
  virtual ~SpVcLLVMGlobalVariableName() = default;
  SpVcLLVMGlobalVariableName(std::string name) : mName(name) {}
  std::string emitIR() override { return "@" + mName; }
};

class SpVcLLVMLabelName : public SpVcLLVMExpr {
  std::string mName;

public:
  virtual ~SpVcLLVMLabelName() = default;
  SpVcLLVMLabelName(std::string name) : mName(name) {}
  std::string emitIR() override { return mName + ":"; }
  std::string emitIRAsVar() { return "%" + mName; }
};

class SpVcLLVMConstantValueInt : public SpVcLLVMVarName {
  int mValue;

public:
  virtual ~SpVcLLVMConstantValueInt() = default;
  SpVcLLVMConstantValueInt(int value) : mValue(value) {}
  std::string emitIR() override { return std::to_string(mValue); }
};

class SpVcLLVMConstantValuePlaceholder : public SpVcLLVMVarName {
public:
  virtual ~SpVcLLVMConstantValuePlaceholder() = default;
  SpVcLLVMConstantValuePlaceholder() {}
  std::string emitIR() override { return ""; }
};

class SpVcLLVMConstantValueFloat32 : public SpVcLLVMVarName {
  float mValue;

private:
  double floatToDoubleDeserialization(float x) {
    auto bin = *(uint32_t *)(&x);
    auto signFloat = bin >> 31;
    auto exponentFloat = (bin >> 23) & (0xFF);
    auto fractionFloat = bin & (0x7FFFFF);
    auto isNan = (fractionFloat != 0) && (exponentFloat == 0xFF);
    auto isInf = (fractionFloat == 0) && (exponentFloat == 0xFF);
    uint64_t exponentDouble = 0;
    uint64_t fractionDouble = 0;
    if (isNan || isInf) {
      exponentDouble = 0x7FF;
      if (isInf)
        fractionDouble = 0;
      else
        fractionDouble = 0x8000000000000;
    } else {
      exponentFloat = exponentFloat + 1023 - 127;
      exponentDouble = exponentFloat;
      fractionDouble = static_cast<uint64_t>(fractionFloat) << (52 - 23);
    }
    uint64_t signDouble = static_cast<uint64_t>(signFloat) << 63;
    exponentDouble = exponentDouble << 52;
    uint64_t result = signDouble | exponentDouble | fractionDouble;
    return *(double *)(&result);
  }

public:
  virtual ~SpVcLLVMConstantValueFloat32() = default;
  SpVcLLVMConstantValueFloat32(float value) : mValue(value) {}
  SpVcLLVMConstantValueFloat32(int value)
      : mValue(std::bit_cast<float, int>(value)) {}

  std::string emitIR() override {
    auto doubleVal = floatToDoubleDeserialization(mValue);
    return std::format("0x{:X}", *(uint64_t *)(&doubleVal));
  }
};

class SpVcTypedVariable
    : public virtual SpVcLLVMExpr,
      public virtual SpVcLLVMCompContainer<SpVcLLVMType *, SpVcLLVMVarName *> {
public:
  virtual ~SpVcTypedVariable() = default;
  SpVcTypedVariable(SpVcLLVMType *type, SpVcLLVMVarName *name)
      : SpVcLLVMCompContainer(type, name) {}
  std::string emitIR() override {
    return std::get<0>(mComponents)->emitIR() + " " +
           std::get<1>(mComponents)->emitIR();
  }
  std::string emitIRType() { return std::get<0>(mComponents)->emitIR(); }
  std::string emitIRName() { return std::get<1>(mComponents)->emitIR(); }
};

using SpVcLLVMArgument = SpVcTypedVariable;

class SpVcLLVMConstantValueVector : public SpVcLLVMVarName {
  std::vector<SpVcLLVMArgument *> mValues;

public:
  virtual ~SpVcLLVMConstantValueVector() = default;
  SpVcLLVMConstantValueVector(std::vector<SpVcLLVMArgument *> values)
      : mValues(values) {
    if (values.size() == 0) {
      ifritError("Empty vector");
    }
  }
  std::string emitIR() override {
    std::string ret = "<";
    for (auto &val : mValues) {
      ret += val->emitIR() + ", ";
    }
    ret.pop_back();
    ret.pop_back();
    ret += ">";
    return ret;
  }
};

class SpVcLLVMConstantValueUndef : public SpVcLLVMVarName {
public:
  virtual ~SpVcLLVMConstantValueUndef() = default;
  SpVcLLVMConstantValueUndef() {}
  std::string emitIR() override { return "undef"; }
};

// Instructions
class SpVcLLVMInstruction : public SpVcLLVMExpr {
public:
  virtual ~SpVcLLVMInstruction() = default;
  virtual std::string emitIR() = 0;
};

class SpVcLLVMIns_TypeAlias
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMLocalVariableName *,
                                   SpVcLLVMType *> {
public:
  virtual ~SpVcLLVMIns_TypeAlias() = default;
  SpVcLLVMIns_TypeAlias(SpVcLLVMLocalVariableName *name, SpVcLLVMType *type)
      : SpVcLLVMCompContainer(name, type) {}
  std::string emitIR() override {
    return std::get<0>(mComponents)->emitIR() + " = type " +
           std::get<1>(mComponents)->emitIR();
  }
};

class SpVcLLVMIns_Alloca
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMType *> {
public:
  virtual ~SpVcLLVMIns_Alloca() = default;
  SpVcLLVMIns_Alloca(SpVcLLVMArgument *dest, SpVcLLVMType *size)
      : SpVcLLVMCompContainer(dest, size) {}
  std::string emitIR() override {
    return std::format("{} = alloca {}", std::get<0>(mComponents)->emitIRName(),
                       std::get<1>(mComponents)->emitIR());
  }
};

class SpVcLLVMIns_Load
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *> {
public:
  virtual ~SpVcLLVMIns_Load() = default;
  SpVcLLVMIns_Load(SpVcLLVMArgument *dest, SpVcLLVMArgument *src)
      : SpVcLLVMCompContainer(dest, src) {}
  std::string emitIR() override {
    return std::format("{} = load {}, {}",
                       std::get<0>(mComponents)->emitIRName(),
                       std::get<0>(mComponents)->emitIRType(),
                       std::get<1>(mComponents)->emitIR());
  }
};

class SpVcLLVMIns_LoadForcedPtr
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *> {
public:
  virtual ~SpVcLLVMIns_LoadForcedPtr() = default;
  SpVcLLVMIns_LoadForcedPtr(SpVcLLVMArgument *dest, SpVcLLVMArgument *src)
      : SpVcLLVMCompContainer(dest, src) {}
  std::string emitIR() override {
    return std::format("{} = load {}, {}* {}",
                       std::get<0>(mComponents)->emitIRName(),
                       std::get<0>(mComponents)->emitIRType(),
                       std::get<1>(mComponents)->emitIRType(),
                       std::get<1>(mComponents)->emitIRName());
  }
};

class SpVcLLVMIns_Note : public SpVcLLVMInstruction {
private:
  std::string st;

public:
  virtual ~SpVcLLVMIns_Note() = default;
  SpVcLLVMIns_Note(std::string x) : st(x) {}
  std::string emitIR() override { return "; " + st; }
};

class SpVcLLVMIns_Store
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *> {
public:
  virtual ~SpVcLLVMIns_Store() = default;
  SpVcLLVMIns_Store(SpVcLLVMArgument *dest, SpVcLLVMArgument *src)
      : SpVcLLVMCompContainer(src, dest) {
    if (dest == nullptr) {
      ifritError("Empty dest");
    }
  }
  std::string emitIR() override {
    return std::format("store {}, {}", std::get<0>(mComponents)->emitIR(),
                       std::get<1>(mComponents)->emitIR());
  }
};

class SpVcLLVMIns_StoreForcedPtr
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *> {
public:
  virtual ~SpVcLLVMIns_StoreForcedPtr() = default;
  SpVcLLVMIns_StoreForcedPtr(SpVcLLVMArgument *dest, SpVcLLVMArgument *src)
      : SpVcLLVMCompContainer(src, dest) {
    if (dest == nullptr) {
      ifritError("Empty dest");
    }
  }
  std::string emitIR() override {
    return std::format("store {}, {}* {}", std::get<0>(mComponents)->emitIR(),
                       std::get<1>(mComponents)->emitIRType(),
                       std::get<1>(mComponents)->emitIRName());
  }
};

class SpVcLLVMIns_GlobalVariable
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *> {
public:
  virtual ~SpVcLLVMIns_GlobalVariable() = default;
  SpVcLLVMIns_GlobalVariable(SpVcLLVMArgument *dest)
      : SpVcLLVMCompContainer(dest) {}
  std::string emitIR() override {
    return std::format("{} = global {}  undef",
                       std::get<0>(mComponents)->emitIRName(),
                       std::get<0>(mComponents)->emitIRType());
  }
};

class SpVcLLVMIns_MathBinary
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *,
                                   SpVcLLVMArgument *> {
  std::string mOp;

public:
  virtual ~SpVcLLVMIns_MathBinary() = default;
  SpVcLLVMIns_MathBinary(SpVcLLVMArgument *dest, SpVcLLVMArgument *src1,
                         SpVcLLVMArgument *src2, std::string op)
      : SpVcLLVMCompContainer(dest, src1, src2), mOp(op) {}
  std::string emitIR() override {
    return std::format("{} = {} {}, {}", std::get<0>(mComponents)->emitIRName(),
                       mOp, std::get<1>(mComponents)->emitIR(),
                       std::get<2>(mComponents)->emitIRName());
  }
};

class SpVcLLVMIns_MathUnary
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *> {
  std::string mOp;

public:
  virtual ~SpVcLLVMIns_MathUnary() = default;
  SpVcLLVMIns_MathUnary(SpVcLLVMArgument *dest, SpVcLLVMArgument *src,
                        std::string op)
      : SpVcLLVMCompContainer(dest, src), mOp(op) {}
  std::string emitIR() override {
    return std::format("{} = {} {}", std::get<0>(mComponents)->emitIRName(),
                       mOp, std::get<1>(mComponents)->emitIR());
  }
};

class SpVcLLVMIns_Ret : public SpVcLLVMInstruction,
                        public SpVcLLVMCompContainer<SpVcLLVMArgument *> {
public:
  virtual ~SpVcLLVMIns_Ret() = default;
  SpVcLLVMIns_Ret(SpVcLLVMArgument *src) : SpVcLLVMCompContainer(src) {}
  std::string emitIR() override {
    return std::format("ret {}", std::get<0>(mComponents)->emitIR());
  }
};

class SpVcLLVMIns_Br : public SpVcLLVMInstruction,
                       public SpVcLLVMCompContainer<SpVcLLVMLabelName *> {
public:
  virtual ~SpVcLLVMIns_Br() = default;

  SpVcLLVMIns_Br(SpVcLLVMLabelName *label) : SpVcLLVMCompContainer(label) {}
  std::string emitIR() override {
    return std::format("br label {}", std::get<0>(mComponents)->emitIRAsVar());
  }
};

class SpVcLLVMIns_BrCond
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMLabelName *,
                                   SpVcLLVMLabelName *> {
public:
  virtual ~SpVcLLVMIns_BrCond() = default;
  SpVcLLVMIns_BrCond(SpVcLLVMArgument *cond, SpVcLLVMLabelName *trueLabel,
                     SpVcLLVMLabelName *falseLabel)
      : SpVcLLVMCompContainer(cond, trueLabel, falseLabel) {
    if (trueLabel == nullptr || falseLabel == nullptr) {
      ifritError("Dest nullptr");
    }
  }
  std::string emitIR() override {
    return std::format("br {}, label {}, label {}",
                       std::get<0>(mComponents)->emitIR(),
                       std::get<1>(mComponents)->emitIRAsVar(),
                       std::get<2>(mComponents)->emitIRAsVar());
  }
};

class SpVcLLVMIns_BeginFunction : public SpVcLLVMInstruction {
  std::string functionName;
  SpVcLLVMType *returnType;
  std::vector<SpVcLLVMArgument *> arguments;

public:
  SpVcLLVMIns_BeginFunction(std::string name, SpVcLLVMType *returnType,
                            std::vector<SpVcLLVMArgument *> args)
      : functionName(name), returnType(returnType), arguments(args) {}
  std::string emitIR() override {
    std::string ret =
        std::format("define {} @{}(", returnType->emitIR(), functionName);
    for (auto &arg : arguments) {
      ret += arg->emitIR() + ", ";
    }
    ret.pop_back();
    ret.pop_back();
    ret += ") {";
    return ret;
  }
};

class SpVcLLVMIns_EndFunction : public SpVcLLVMInstruction {
public:
  std::string emitIR() override { return "}"; }
};

class SpVcLLVMIns_Phi : public SpVcLLVMInstruction {
private:
  std::vector<std::pair<SpVcLLVMArgument *, SpVcLLVMLabelName *>> p;
  SpVcLLVMArgument *retv;

public:
  SpVcLLVMIns_Phi(
      SpVcLLVMArgument *res,
      std::vector<std::pair<SpVcLLVMArgument *, SpVcLLVMLabelName *>> conds) {
    p = conds;
    retv = res;
  }
  std::string emitIR() override {
    std::string ret;
    ret = retv->emitIR() + " = phi ";
    for (auto &[arg, label] : p) {
      ret += "[" + arg->emitIR() + ", " + label->emitIR() + "], ";
    }
    ret.pop_back();
    ret.pop_back();
    return ret + "\n";
  }
};

class SpVcLLVMIns_ImmediateAssign
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *> {
public:
  SpVcLLVMIns_ImmediateAssign(SpVcLLVMArgument *dest, SpVcLLVMArgument *src)
      : SpVcLLVMCompContainer(dest, src) {}
  std::string emitIR() override {
    // LLVM does not support direct constant assignment, but there are indirect
    // ways like %x = add i32 1,0
    return std::format("{} = add {}, 0", std::get<0>(mComponents)->emitIR(),
                       std::get<1>(mComponents)->emitIR());
  }
};
class SpVcLLVMIns_Select
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *,
                                   SpVcLLVMArgument *, SpVcLLVMArgument *> {
public:
  SpVcLLVMIns_Select(SpVcLLVMArgument *dest, SpVcLLVMArgument *cond,
                     SpVcLLVMArgument *trueVal, SpVcLLVMArgument *falseVal)
      : SpVcLLVMCompContainer(dest, cond, trueVal, falseVal) {}
  std::string emitIR() override {
    return std::format(
        "{} = select {}, {}, {}", std::get<0>(mComponents)->emitIRName(),
        std::get<1>(mComponents)->emitIR(), std::get<2>(mComponents)->emitIR(),
        std::get<3>(mComponents)->emitIR());
  }
};

class SpVcLLVMIns_InsertElement
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *,
                                   SpVcLLVMArgument *, SpVcLLVMArgument *> {
public:
  SpVcLLVMIns_InsertElement(SpVcLLVMArgument *dest, SpVcLLVMArgument *vec,
                            SpVcLLVMArgument *val, SpVcLLVMArgument *idx)
      : SpVcLLVMCompContainer(dest, vec, val, idx) {}
  std::string emitIR() override {
    return std::format(
        "{} = insertelement {}, {}, {}", std::get<0>(mComponents)->emitIRName(),
        std::get<1>(mComponents)->emitIR(), std::get<2>(mComponents)->emitIR(),
        std::get<3>(mComponents)->emitIR());
  }
};

class SpVcLLVMIns_InsertElementWithConstantIndex
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *,
                                   SpVcLLVMArgument *, int> {
public:
  SpVcLLVMIns_InsertElementWithConstantIndex(SpVcLLVMArgument *dest,
                                             SpVcLLVMArgument *vec,
                                             SpVcLLVMArgument *val, int idx)
      : SpVcLLVMCompContainer(dest, vec, val, idx) {
    if (vec == nullptr) {
      ifritError("Empty vec");
    }
  }
  std::string emitIR() override {
    return std::format("{} = insertelement {}, {}, i32 {}",
                       std::get<0>(mComponents)->emitIRName(),
                       std::get<1>(mComponents)->emitIR(),
                       std::get<2>(mComponents)->emitIR(),
                       std::get<3>(mComponents));
  }
};
class SpVcLLVMIns_InsertElementWithConstantIndexUndefInit
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *,
                                   int> {
public:
  SpVcLLVMIns_InsertElementWithConstantIndexUndefInit(SpVcLLVMArgument *dest,
                                                      SpVcLLVMArgument *vec,
                                                      int idx)
      : SpVcLLVMCompContainer(dest, vec, idx) {}
  std::string emitIR() override {
    return std::format("{} = insertelement {} undef, {} , i32 {}",
                       std::get<0>(mComponents)->emitIRName(),
                       std::get<0>(mComponents)->emitIRType(),
                       std::get<1>(mComponents)->emitIR(),
                       std::get<2>(mComponents));
  }
};

class SpVcLLVMIns_ExtractElementWithConstantIndex
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *,
                                   int> {
public:
  SpVcLLVMIns_ExtractElementWithConstantIndex(SpVcLLVMArgument *dest,
                                              SpVcLLVMArgument *vec, int idx)
      : SpVcLLVMCompContainer(dest, vec, idx) {}
  std::string emitIR() override {
    return std::format("{} = extractelement {}, i32 {}",
                       std::get<0>(mComponents)->emitIRName(),
                       std::get<1>(mComponents)->emitIR(),
                       std::get<2>(mComponents));
  }
};

class SpVcLLVMIns_FunctionCallExtInst : public SpVcLLVMInstruction {
  SpVcLLVMArgument *retVal;
  std::string mName;
  std::vector<SpVcLLVMArgument *> mArgs;

public:
  virtual ~SpVcLLVMIns_FunctionCallExtInst() = default;
  SpVcLLVMIns_FunctionCallExtInst(SpVcLLVMArgument *retVal,
                                  std::string funcName,
                                  const std::vector<SpVcLLVMArgument *> args)
      : retVal(retVal), mName(funcName), mArgs(args) {}
  std::string emitIR() override {
    std::string ret = std::format("{} = call {} @{}(", retVal->emitIRName(),
                                  retVal->emitIRType(), mName);
    for (auto &arg : mArgs) {
      ret += arg->emitIR() + ", ";
    }
    ret.pop_back();
    ret.pop_back();
    ret += ")";
    return ret;
  }
};

class SpVcLLVMIns_FunctionFragmentEntry : public SpVcLLVMInstruction {
  std::string mName;

public:
  SpVcLLVMIns_FunctionFragmentEntry(std::string name) : mName(name) {}
  std::string emitIR() override {
    return std::format("define void @{}", mName) + "() #0 {";
  }
};

class SpVcLLVMIns_FunctionEnd : public SpVcLLVMInstruction {
public:
  std::string emitIR() override { return "}"; }
};

class SpVcLLVMIns_ExtractValueWithConstantIndex
    : public SpVcLLVMInstruction,
      public SpVcLLVMCompContainer<SpVcLLVMArgument *, SpVcLLVMArgument *,
                                   int> {
public:
  SpVcLLVMIns_ExtractValueWithConstantIndex(SpVcLLVMArgument *dest,
                                            SpVcLLVMArgument *vec, int idx)
      : SpVcLLVMCompContainer(dest, vec, idx) {}
  std::string emitIR() override {
    return std::format(
        "{} = extractvalue {}, {}", std::get<0>(mComponents)->emitIRName(),
        std::get<1>(mComponents)->emitIR(), std::get<2>(mComponents));
  }
};

class SpVcLLVMIns_ReturnVoid : public SpVcLLVMInstruction {
public:
  std::string emitIR() override { return "ret void"; }
};

} // namespace
  // Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec::LLVM