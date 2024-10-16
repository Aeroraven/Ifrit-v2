#pragma once
#include "SpvMdBase.h"
#include "SpvMdQuadIRGenerator.h"
#include "engine/base/ShaderRuntime.h"
#include "engine/base/Shaders.h"
#include "engine/shadervm/spirv/SpvVMReader.h"

namespace Ifrit::Engine::SoftRenderer::ShaderVM::SpirvVec {
struct SpvVecRuntimeSymbolTables {
  std::vector<void *> inputs[SpVcQuadSize];
  std::vector<int> inputBytes[SpVcQuadSize];
  std::vector<void *> outputs[SpVcQuadSize];
  std::vector<int> outputBytes[SpVcQuadSize];
  std::unordered_map<std::pair<int, int>, std::pair<void *, int>,
                     Ifrit::Engine::SoftRenderer::Core::Utility::PairHash>
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
} // namespace Ifrit::Engine::SoftRenderer::ShaderVM::SpirvVec