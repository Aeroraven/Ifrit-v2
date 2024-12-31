
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


#pragma once
#include "SpvMdBase.h"
#include "SpvMdQuadIRGenerator.h"
#include "ifrit/softgraphics/engine/base/ShaderRuntime.h"
#include "ifrit/softgraphics/engine/base/Shaders.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMReader.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec {
struct SpvVecRuntimeSymbolTables {
  std::vector<void *> inputs[SpVcQuadSize];
  std::vector<int> inputBytes[SpVcQuadSize];
  std::vector<void *> outputs[SpVcQuadSize];
  std::vector<int> outputBytes[SpVcQuadSize];
  std::unordered_map<
      std::pair<int, int>, std::pair<void *, int>,
      Ifrit::GraphicsBackend::SoftGraphics::Core::Utility::PairHash>
      uniform;
  void *entry = nullptr;
};

class SpvVecRuntimeBackend {
protected:
  static int createTime;
  Spirv::SpvVMReader reader;
  SpVcQuadGroupedIRGenerator interpreter;
  Spirv::SpvVMContext spctx;
  SpVcVMGeneratorContext spvir;
  const SpVcVMGeneratorContext *spvirRef;
  ShaderRuntime *runtime;
  SpvVecRuntimeSymbolTables symbolTables;
  std::unique_ptr<ShaderRuntime> copiedRuntime = nullptr;
  std::string irCode;

  std::unique_ptr<ShaderRuntime> owningRuntime = nullptr;

public:
  SpvVecRuntimeBackend(const ShaderRuntimeBuilder &runtime,
                       std::vector<char> irByteCode);
  SpvVecRuntimeBackend(const SpvVecRuntimeBackend &other);

protected:
  void updateSymbolTable(bool isCopy);
};

class SpvVecFragmentShader final : public FragmentShader,
                                   public SpvVecRuntimeBackend {
public:
  SpvVecFragmentShader(const SpvVecFragmentShader &p);

public:
  SpvVecFragmentShader(const ShaderRuntimeBuilder &runtime,
                       std::vector<char> irByteCode);
  ~SpvVecFragmentShader() = default;
  IFRIT_DUAL virtual void execute(const void *varyings, void *colorOutput,
                                  float *fragmentDepth) override;
  IFRIT_HOST virtual void executeInQuad(const void **varyings,
                                        void **colorOutput,
                                        float **fragmentDepth) override;
  IFRIT_HOST virtual FragmentShader *getCudaClone() override;
  IFRIT_HOST virtual std::unique_ptr<FragmentShader>
  getThreadLocalCopy() override;
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) override;
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::SpirvVec