
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

#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMShader.h"
#include "ifrit/common/util/TypingUtil.h"
using namespace Ifrit::Common::Utility;
extern "C" {
struct alignas(16) Vector3iAligned {
  int x, y, z;
};
}

namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::Spirv {
int SpvRuntimeBackend::createTime = 0;
SpvRuntimeBackend::SpvRuntimeBackend(const ShaderRuntimeBuilder &runtime,
                                     std::vector<char> irByteCode) {
  reader.initializeContext(&spctx);
  reader.parseByteCode(irByteCode.data(), irByteCode.size() / 4, &spctx);
  interpreter.parseRawContext(&spctx, &spvir);
  this->owningRuntime = runtime.buildRuntime();
  this->runtime = owningRuntime.get();
  interpreter.exportLlvmIR(&spvir, &this->irCode);
  this->runtime->loadIR(this->irCode, std::to_string(this->createTime));
  updateSymbolTable(false);
  ifritLog1("Shader loaded");
}
SpvRuntimeBackend::SpvRuntimeBackend(const SpvRuntimeBackend &other) {
  this->copiedRuntime = other.runtime->getThreadLocalCopy();
  this->runtime = this->copiedRuntime.get();
  this->irCode = other.irCode;
  this->spvirRef = &other.spvir;
  updateSymbolTable(true);
}
void SpvRuntimeBackend::updateSymbolTable(bool isCopy) {
  const SpvVMIntermediateRepresentation *svmir = (isCopy) ? spvirRef : &spvir;

  this->symbolTables.inputs.resize(svmir->shaderMaps.inputVarSymbols.size());
  this->symbolTables.outputs.resize(svmir->shaderMaps.outputVarSymbols.size());
  this->symbolTables.inputBytes.resize(
      svmir->shaderMaps.inputVarSymbols.size());
  this->symbolTables.outputBytes.resize(
      svmir->shaderMaps.outputVarSymbols.size());
  for (int i = 0; i < svmir->shaderMaps.inputVarSymbols.size(); i++) {
    this->symbolTables.inputs[i] =
        this->runtime->lookupSymbol(svmir->shaderMaps.inputVarSymbols[i]);
    this->symbolTables.inputBytes[i] = svmir->shaderMaps.inputSize[i];
  }
  for (int i = 0; i < svmir->shaderMaps.outputVarSymbols.size(); i++) {
    this->symbolTables.outputs[i] =
        this->runtime->lookupSymbol(svmir->shaderMaps.outputVarSymbols[i]);
    this->symbolTables.outputBytes[i] = svmir->shaderMaps.outputSize[i];
  }
  cSISize = size_cast<int>(svmir->shaderMaps.inputVarSymbols.size());
  cSOSize = size_cast<int>(svmir->shaderMaps.outputVarSymbols.size());
  for (int i = 0; i < svmir->shaderMaps.uniformSize.size(); i++) {
    std::pair<int, int> loc = svmir->shaderMaps.uniformVarLoc[i];
    this->symbolTables.uniform[loc] = {
        this->runtime->lookupSymbol(svmir->shaderMaps.uniformVarSymbols[i]),
        svmir->shaderMaps.uniformSize[i]};
  }
  this->symbolTables.entry =
      this->runtime->lookupSymbol(svmir->shaderMaps.mainFuncSymbol);
  cEntry = (void (*)())this->symbolTables.entry;
  if (svmir->shaderMaps.builtinPositionSymbol.size()) {
    this->symbolTables.builtinPosition =
        this->runtime->lookupSymbol(svmir->shaderMaps.builtinPositionSymbol);
  }
  if (svmir->shaderMaps.builtinLaunchIdKHR.size()) {
    this->symbolTables.builtinLaunchId =
        this->runtime->lookupSymbol(svmir->shaderMaps.builtinLaunchIdKHR);
  }
  if (svmir->shaderMaps.builtinLaunchSizeKHR.size()) {
    this->symbolTables.builtinLaunchSize =
        this->runtime->lookupSymbol(svmir->shaderMaps.builtinLaunchSizeKHR);
  }
  if (svmir->shaderMaps.incomingRayPayloadKHR.size()) {
    this->symbolTables.incomingPayload =
        this->runtime->lookupSymbol(svmir->shaderMaps.incomingRayPayloadKHR);
    this->symbolTables.incomingPayloadSize =
        svmir->shaderMaps.incomingRayPayloadKHRSize;
  }
  this->symbolTables.builtinContext =
      this->runtime->lookupSymbol("ifsp_builtin_context_ptr");
}

SpvVertexShader::SpvVertexShader(const ShaderRuntimeBuilder &runtime,
                                 std::vector<char> irByteCode)
    : SpvRuntimeBackend(runtime, irByteCode) {
  isThreadSafe = false;
}
SpvVertexShader::SpvVertexShader(const SpvVertexShader &p)
    : SpvRuntimeBackend(p) {
  isThreadSafe = false;
}
IFRIT_HOST std::unique_ptr<VertexShader> SpvVertexShader::getThreadLocalCopy() {
  auto copy = std::make_unique<SpvVertexShader>(*this);
  return copy;
}

IFRIT_HOST void SpvVertexShader::updateUniformData(int binding, int set,
                                                   const void *pData) {
  auto &uniformData = symbolTables.uniform[{binding, set}];
  memcpy(uniformData.first, pData, uniformData.second);
}

IFRIT_HOST std::vector<std::pair<int, int>> SpvVertexShader::getUniformList() {
  std::vector<std::pair<int, int>> ret;
  for (auto &p : this->symbolTables.uniform) {
    ret.push_back(p.first);
  }
  return ret;
}

IFRIT_HOST VaryingDescriptor SpvVertexShader::getVaryingDescriptor() {
  VaryingDescriptor vdesc;
  std::vector<TypeDescriptor> tpDesc{};
  for (int i = 0; i < symbolTables.outputs.size(); i++) {
    tpDesc.push_back(TypeDescriptors.FLOAT4);
  }
  vdesc.setVaryingDescriptors(tpDesc);
  return vdesc;
}

SpvFragmentShader::SpvFragmentShader(const ShaderRuntimeBuilder &runtime,
                                     std::vector<char> irByteCode)
    : SpvRuntimeBackend(runtime, irByteCode) {
  isThreadSafe = false;
}
SpvFragmentShader::SpvFragmentShader(const SpvFragmentShader &p)
    : SpvRuntimeBackend(p) {
  isThreadSafe = false;
  this->allowDepthModification = p.allowDepthModification;
  this->requiresQuadInfo = p.requiresQuadInfo;
}
IFRIT_HOST std::unique_ptr<FragmentShader>
SpvFragmentShader::getThreadLocalCopy() {
  auto copy = std::make_unique<SpvFragmentShader>(*this);
  return copy;
}

IFRIT_HOST void SpvFragmentShader::updateUniformData(int binding, int set,
                                                     const void *pData) {
  auto &uniformData = symbolTables.uniform[{binding, set}];
  memcpy(uniformData.first, pData, uniformData.second);
}

void SpvVertexShader::execute(const void *const *input, Vector4f *outPos,
                              Vector4f *const *outVaryings) {
  // TODO: Input & Output
  auto &sI = symbolTables.inputs;
  auto &sO = symbolTables.outputs;
  auto &sOb = symbolTables.outputBytes;
  auto &sIb = symbolTables.inputBytes;
  auto sISize = cSISize;
  auto sOSize = cSOSize;
  for (int i = 0; i < sISize; i++) {
    memcpy(sI[i], input[i], sIb[i]);
  }
  cEntry();
  for (int i = 0; i < sOSize; i++) {
    memcpy(outVaryings[i], sO[i], sOb[i]);
  }
  auto ptrPos = (Vector4f *)symbolTables.builtinPosition;
  if (ptrPos) {
    *outPos = *ptrPos;
  }
}
IFRIT_HOST VertexShader *SpvVertexShader::getCudaClone() {
  ifritError("CUDA not supported");
  return nullptr;
}
void SpvFragmentShader::execute(const void *varyings, void *colorOutput,
                                float *fragmentDepth) {
  // TODO: Input & Output
  auto &sI = symbolTables.inputs;
  auto &sO = symbolTables.outputs;
  auto sISize = cSISize;
  auto sOSize = cSOSize;
  for (int i = 0; i < sISize; i++) {
    *((VaryingStore *)sI[i]) = *((VaryingStore *)varyings + i);
  }
  cEntry();
  for (int i = 0; i < sOSize; i++) {
    *((Vector4f *)colorOutput + i) = *((Vector4f *)sO[i]);
  }
}
IFRIT_HOST FragmentShader *SpvFragmentShader::getCudaClone() {
  ifritError("CUDA not supported");
  return nullptr;
}

IFRIT_HOST std::vector<std::pair<int, int>>
SpvFragmentShader::getUniformList() {
  std::vector<std::pair<int, int>> ret;
  for (auto &p : this->symbolTables.uniform) {
    ret.push_back(p.first);
  }
  return ret;
}

SpvRaygenShader::SpvRaygenShader(const SpvRaygenShader &p)
    : SpvRuntimeBackend(p) {
  isThreadSafe = false;
}
SpvRaygenShader::SpvRaygenShader(const ShaderRuntimeBuilder &runtime,
                                 std::vector<char> irByteCode)
    : SpvRuntimeBackend(runtime, irByteCode) {
  isThreadSafe = false;
}
IFRIT_DUAL void SpvRaygenShader::execute(const Vector3i &inputInvocation,
                                         const Vector3i &dimension,
                                         void *context) {
  if (symbolTables.builtinLaunchId)
    memcpy(symbolTables.builtinLaunchId, &inputInvocation, sizeof(Vector3i));
  if (symbolTables.builtinLaunchSize)
    memcpy(symbolTables.builtinLaunchSize, &dimension, sizeof(Vector3i));
  if (symbolTables.builtinContext)
    memcpy(symbolTables.builtinContext, &context, sizeof(void *));
  auto shaderEntry = (void (*)())this->symbolTables.entry;
  // checkAddress(shaderEntry);
  shaderEntry();
}
IFRIT_HOST Raytracer::RayGenShader *SpvRaygenShader::getCudaClone() {
  ifritError("CUDA not supported");
  return nullptr;
}
IFRIT_HOST std::unique_ptr<Raytracer::RayGenShader>
SpvRaygenShader::getThreadLocalCopy() {
  auto copy = std::make_unique<SpvRaygenShader>(*this);
  return copy;
}
IFRIT_HOST void SpvRaygenShader::updateUniformData(int binding, int set,
                                                   const void *pData) {
  auto &uniformData = symbolTables.uniform[{binding, set}];
  memcpy(uniformData.first, pData, uniformData.second);
}
IFRIT_HOST std::vector<std::pair<int, int>> SpvRaygenShader::getUniformList() {
  std::vector<std::pair<int, int>> ret;
  for (auto &p : this->symbolTables.uniform) {
    ret.push_back(p.first);
  }
  return ret;
}

SpvMissShader::SpvMissShader(const SpvMissShader &p) : SpvRuntimeBackend(p) {
  isThreadSafe = false;
}
IFRIT_HOST void SpvMissShader::updateStack() {
  if (this->execStack.size() == 0)
    return;
  const auto &stackTop = this->execStack.back();
  if (symbolTables.incomingPayload) {
    memcpy(symbolTables.incomingPayload, stackTop.payloadPtr,
           symbolTables.incomingPayloadSize);
  }
}
SpvMissShader::SpvMissShader(const ShaderRuntimeBuilder &runtime,
                             std::vector<char> irByteCode)
    : SpvRuntimeBackend(runtime, irByteCode) {
  isThreadSafe = false;
}
IFRIT_DUAL void SpvMissShader::execute(void *context) {
  if (symbolTables.builtinContext)
    memcpy(symbolTables.builtinContext, &context, sizeof(void *));
  auto shaderEntry = (void (*)())this->symbolTables.entry;
  shaderEntry();

  // Update payloads
  const auto &stackTop = this->execStack.back();
  if (symbolTables.incomingPayload) {
    memcpy(stackTop.payloadPtr, symbolTables.incomingPayload,
           symbolTables.incomingPayloadSize);
  }
}
IFRIT_HOST Raytracer::MissShader *SpvMissShader::getCudaClone() {
  ifritError("CUDA not supported");
  return nullptr;
}
IFRIT_HOST std::unique_ptr<Raytracer::MissShader>
SpvMissShader::getThreadLocalCopy() {
  auto copy = std::make_unique<SpvMissShader>(*this);
  return copy;
}
IFRIT_HOST void SpvMissShader::updateUniformData(int binding, int set,
                                                 const void *pData) {
  auto &uniformData = symbolTables.uniform[{binding, set}];
  memcpy(uniformData.first, pData, uniformData.second);
}
IFRIT_HOST std::vector<std::pair<int, int>> SpvMissShader::getUniformList() {
  std::vector<std::pair<int, int>> ret;
  for (auto &p : this->symbolTables.uniform) {
    ret.push_back(p.first);
  }
  return ret;
}
IFRIT_HOST void SpvMissShader::onStackPushComplete() { updateStack(); }
IFRIT_HOST void SpvMissShader::onStackPopComplete() { updateStack(); }
SpvClosestHitShader::SpvClosestHitShader(const SpvClosestHitShader &p)
    : SpvRuntimeBackend(p) {
  isThreadSafe = false;
}
IFRIT_HOST void SpvClosestHitShader::updateStack() {
  if (this->execStack.size() == 0)
    return;
  const auto &stackTop = this->execStack.back();
  if (symbolTables.incomingPayload) {
    memcpy(symbolTables.incomingPayload, stackTop.payloadPtr,
           symbolTables.incomingPayloadSize);
  }
}
SpvClosestHitShader::SpvClosestHitShader(const ShaderRuntimeBuilder &runtime,
                                         std::vector<char> irByteCode)
    : SpvRuntimeBackend(runtime, irByteCode) {
  isThreadSafe = false;
}
IFRIT_DUAL void SpvClosestHitShader::execute(const RayHit &hitAttribute,
                                             const RayInternal &ray,
                                             void *context) {
  if (symbolTables.builtinContext)
    memcpy(symbolTables.builtinContext, &context, sizeof(void *));
  auto shaderEntry = (void (*)())this->symbolTables.entry;
  shaderEntry();

  // Update payloads
  const auto &stackTop = this->execStack.back();
  if (symbolTables.incomingPayload) {
    memcpy(stackTop.payloadPtr, symbolTables.incomingPayload,
           symbolTables.incomingPayloadSize);
  }
}
IFRIT_HOST Raytracer::CloseHitShader *SpvClosestHitShader::getCudaClone() {
  ifritError("CUDA not supported");
  return nullptr;
}
IFRIT_HOST std::unique_ptr<Raytracer::CloseHitShader>
SpvClosestHitShader::getThreadLocalCopy() {
  auto copy = std::make_unique<SpvClosestHitShader>(*this);
  return copy;
}
IFRIT_HOST void SpvClosestHitShader::updateUniformData(int binding, int set,
                                                       const void *pData) {
  auto &uniformData = symbolTables.uniform[{binding, set}];
  memcpy(uniformData.first, pData, uniformData.second);
}
IFRIT_HOST std::vector<std::pair<int, int>>
SpvClosestHitShader::getUniformList() {
  std::vector<std::pair<int, int>> ret;
  for (auto &p : this->symbolTables.uniform) {
    ret.push_back(p.first);
  }
  return ret;
}
IFRIT_HOST void SpvClosestHitShader::onStackPushComplete() { updateStack(); }
IFRIT_HOST void SpvClosestHitShader::onStackPopComplete() { updateStack(); }
} // namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::Spirv