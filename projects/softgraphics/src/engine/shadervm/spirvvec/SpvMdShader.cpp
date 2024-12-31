
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#include "ifrit/softgraphics/engine/shadervm/spirvvec/SpvMdShader.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec {
int SpvVecRuntimeBackend::createTime = 0;
SpvVecRuntimeBackend::SpvVecRuntimeBackend(const ShaderRuntimeBuilder &runtime,
                                           std::vector<char> irByteCode) {
  reader.initializeContext(&spctx);
  reader.parseByteCode(irByteCode.data(), irByteCode.size() / 4, &spctx);
  interpreter.bindBytecode(&spctx, &spvir);
  interpreter.init();
  interpreter.parse();
  this->owningRuntime = runtime.buildRuntime();
  this->runtime = owningRuntime.get();
  this->irCode = interpreter.generateIR();
  this->runtime->loadIR(this->irCode, std::to_string(this->createTime));
  ifritLog1("IR loaded");
  printf("%s\n", this->irCode.c_str());
  updateSymbolTable(false);
  ifritLog1("Shader loaded");
}
SpvVecRuntimeBackend::SpvVecRuntimeBackend(const SpvVecRuntimeBackend &other) {
  this->copiedRuntime = other.runtime->getThreadLocalCopy();
  this->runtime = this->copiedRuntime.get();
  this->irCode = other.irCode;
  this->spvirRef = &other.spvir;
  updateSymbolTable(true);
}
void SpvVecRuntimeBackend::updateSymbolTable(bool isCopy) {
  const SpVcVMGeneratorContext *svmir = (isCopy) ? spvirRef : &spvir;

  for (int i = 0; i < SpVcQuadSize; i++) {
    this->symbolTables.inputs[i].resize(svmir->binds.inputVarSymbols[i].size());
    this->symbolTables.outputs[i].resize(
        svmir->binds.outputVarSymbols[i].size());
    this->symbolTables.inputBytes[i].resize(
        svmir->binds.inputVarSymbols[i].size());
    this->symbolTables.outputBytes[i].resize(
        svmir->binds.outputVarSymbols[i].size());
  }
  for (int T = 0; T < SpVcQuadSize; T++) {
    for (int i = 0; i < svmir->binds.inputVarSymbols[T].size(); i++) {
      this->symbolTables.inputs[T][i] =
          this->runtime->lookupSymbol(svmir->binds.inputVarSymbols[T][i]);
      this->symbolTables.inputBytes[T][i] = svmir->binds.inputSize[T][i];
    }
    for (int i = 0; i < svmir->binds.outputVarSymbols[T].size(); i++) {
      this->symbolTables.outputs[T][i] =
          this->runtime->lookupSymbol(svmir->binds.outputVarSymbols[T][i]);
      this->symbolTables.outputBytes[T][i] = svmir->binds.outputSize[T][i];
    }
  }
  for (int i = 0; i < svmir->binds.uniformVarSz.size(); i++) {
    std::pair<int, int> loc = svmir->binds.uniformVarLoc[i];
    this->symbolTables.uniform[loc] = {
        this->runtime->lookupSymbol(svmir->binds.uniformVarSymbols[i]),
        svmir->binds.uniformVarSz[i]};
  }
  this->symbolTables.entry =
      this->runtime->lookupSymbol(svmir->binds.mainFunction);
}

SpvVecFragmentShader::SpvVecFragmentShader(const ShaderRuntimeBuilder &runtime,
                                           std::vector<char> irByteCode)
    : SpvVecRuntimeBackend(runtime, irByteCode) {
  isThreadSafe = false;
  forcedQuadInvocation = true;
}
SpvVecFragmentShader::SpvVecFragmentShader(const SpvVecFragmentShader &p)
    : SpvVecRuntimeBackend(p) {
  isThreadSafe = false;
  forcedQuadInvocation = true;
  this->allowDepthModification = p.allowDepthModification;
  this->requiresQuadInfo = p.requiresQuadInfo;
}
IFRIT_HOST std::unique_ptr<FragmentShader>
SpvVecFragmentShader::getThreadLocalCopy() {
  auto copy = std::make_unique<SpvVecFragmentShader>(*this);
  return copy;
}

IFRIT_HOST void SpvVecFragmentShader::updateUniformData(int binding, int set,
                                                        const void *pData) {
  auto &uniformData = symbolTables.uniform[{binding, set}];
  memcpy(uniformData.first, pData, uniformData.second);
}

void SpvVecFragmentShader::execute(const void *varyings, void *colorOutput,
                                   float *fragmentDepth) {
  ifritError("Shader execution should be organized in quads");
}
IFRIT_HOST void SpvVecFragmentShader::executeInQuad(const void **varyings,
                                                    void **colorOutput,
                                                    float **fragmentDepth) {
  // Here we assume all the inputs are float4
  auto &sIb = symbolTables.inputBytes;
  auto &sI = symbolTables.inputs;
  auto &sO = symbolTables.outputs;
  auto &sOb = symbolTables.outputBytes;
  auto sISize = sIb[0].size();
  auto sOSize = sOb[0].size();
  for (int T = 0; T < SpVcQuadSize; T++) {
    auto ptrSCT = (VaryingStore *)varyings[T];
    auto &refSIT = sI[T];
    for (int i = 0; i < sISize; i++) {
      auto sA = refSIT[i];
      auto sC = ptrSCT + i;
      *((ifloat4 *)sA) = *((ifloat4 *)sC);
    }
  }
  auto shaderEntry = (void (*)())this->symbolTables.entry;
  shaderEntry();
  for (int T = 0; T < SpVcQuadSize; T++) {
    auto ptrColorOutputT = (ifloat4 *)colorOutput[T];
    auto &refSOT = sO[T];
    for (int i = 0; i < sOSize; i++) {
      *(ptrColorOutputT + i) = *((ifloat4 *)refSOT[i]);
    }
  }
}
IFRIT_HOST FragmentShader *SpvVecFragmentShader::getCudaClone() {
  ifritError("CUDA not supported");
  return nullptr;
}

IFRIT_HOST std::vector<std::pair<int, int>>
SpvVecFragmentShader::getUniformList() {
  std::vector<std::pair<int, int>> ret;
  for (auto &p : this->symbolTables.uniform) {
    ret.push_back(p.first);
  }
  return ret;
}
} // namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec