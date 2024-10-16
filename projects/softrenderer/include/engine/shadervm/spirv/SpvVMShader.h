#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/ShaderRuntime.h"
#include "./engine/base/Shaders.h"
#include "./engine/raytracer/RtShaders.h"
#include "./engine/shadervm/spirv/SpvVMInterpreter.h"
#include "./engine/shadervm/spirv/SpvVMReader.h"

namespace Ifrit::Engine::SoftRenderer::ShaderVM::Spirv {
struct SpvRuntimeSymbolTables {
  std::vector<void *> inputs;
  std::vector<int> inputBytes;
  std::vector<void *> outputs;
  std::vector<int> outputBytes;
  std::unordered_map<std::pair<int, int>, std::pair<void *, int>,
                     Ifrit::Engine::SoftRenderer::Core::Utility::PairHash>
      uniform;
  void *entry = nullptr;
  void *builtinPosition = nullptr;
  void *builtinLaunchId = nullptr;
  void *builtinLaunchSize = nullptr;

  void *builtinContext = nullptr;
  void *incomingPayload = nullptr;
  int incomingPayloadSize = 0;
};
class SpvRuntimeBackend {
protected:
  static int createTime;
  SpvVMReader reader;
  SpvVMInterpreter interpreter;
  SpvVMContext spctx;
  SpvVMIntermediateRepresentation spvir;
  const SpvVMIntermediateRepresentation *spvirRef;
  ShaderRuntime *runtime;
  SpvRuntimeSymbolTables symbolTables;
  std::unique_ptr<ShaderRuntime> copiedRuntime = nullptr;
  std::string irCode;

  std::unique_ptr<ShaderRuntime> owningRuntime = nullptr;

  // MinGW does not directly store the size of vector
  // it calculates the size of vector by subtracting the address of the first
  // element from the address of the last elements
  int cSISize = 0;
  int cSOSize = 0;
  void (*cEntry)() = nullptr;

public:
  SpvRuntimeBackend(const ShaderRuntimeBuilder &runtime,
                    std::vector<char> irByteCode);
  SpvRuntimeBackend(const SpvRuntimeBackend &other);

protected:
  void updateSymbolTable(bool isCopy);
};

class SpvVertexShader final : public VertexShader, public SpvRuntimeBackend {
public:
  SpvVertexShader(const SpvVertexShader &p);

public:
  SpvVertexShader(const ShaderRuntimeBuilder &runtime,
                  std::vector<char> irByteCode);
  ~SpvVertexShader() = default;
  IFRIT_DUAL virtual void execute(const void *const *input, ifloat4 *outPos,
                                  ifloat4 *const *outVaryings) override;
  IFRIT_HOST virtual VertexShader *getCudaClone() override;
  IFRIT_HOST virtual std::unique_ptr<VertexShader>
  getThreadLocalCopy() override;
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) override;
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
  IFRIT_HOST virtual VaryingDescriptor getVaryingDescriptor() override;
};

class SpvFragmentShader final : public FragmentShader,
                                public SpvRuntimeBackend {
public:
  SpvFragmentShader(const SpvFragmentShader &p);

public:
  SpvFragmentShader(const ShaderRuntimeBuilder &runtime,
                    std::vector<char> irByteCode);
  ~SpvFragmentShader() = default;
  IFRIT_DUAL virtual void execute(const void *varyings, void *colorOutput,
                                  float *fragmentDepth) override;
  IFRIT_HOST virtual FragmentShader *getCudaClone() override;
  IFRIT_HOST virtual std::unique_ptr<FragmentShader>
  getThreadLocalCopy() override;
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) override;
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
};

// V2
class SpvRaygenShader final : public Raytracer::RayGenShader,
                              public SpvRuntimeBackend {
public:
  SpvRaygenShader(const SpvRaygenShader &p);

public:
  SpvRaygenShader(const ShaderRuntimeBuilder &runtime,
                  std::vector<char> irByteCode);
  ~SpvRaygenShader() = default;
  IFRIT_DUAL virtual void execute(const iint3 &inputInvocation,
                                  const iint3 &dimension,
                                  void *context) override;
  IFRIT_HOST virtual Raytracer::RayGenShader *getCudaClone() override;
  IFRIT_HOST virtual std::unique_ptr<Raytracer::RayGenShader>
  getThreadLocalCopy() override;
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) override;
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
};

class SpvMissShader final : public Raytracer::MissShader,
                            public SpvRuntimeBackend {
public:
  SpvMissShader(const SpvMissShader &p);

private:
  IFRIT_HOST void updateStack();

public:
  SpvMissShader(const ShaderRuntimeBuilder &runtime,
                std::vector<char> irByteCode);
  ~SpvMissShader() = default;
  IFRIT_DUAL virtual void execute(void *context) override;
  IFRIT_HOST virtual Raytracer::MissShader *getCudaClone() override;
  IFRIT_HOST virtual std::unique_ptr<Raytracer::MissShader>
  getThreadLocalCopy() override;
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) override;
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
  IFRIT_HOST virtual void onStackPushComplete() override;
  IFRIT_HOST virtual void onStackPopComplete() override;
};

class SpvClosestHitShader final : public Raytracer::CloseHitShader,
                                  public SpvRuntimeBackend {
public:
  SpvClosestHitShader(const SpvClosestHitShader &p);

private:
  IFRIT_HOST void updateStack();

public:
  SpvClosestHitShader(const ShaderRuntimeBuilder &runtime,
                      std::vector<char> irByteCode);
  ~SpvClosestHitShader() = default;
  IFRIT_DUAL virtual void execute(const RayHit &hitAttribute,
                                  const RayInternal &ray,
                                  void *context) override;
  IFRIT_HOST virtual Raytracer::CloseHitShader *getCudaClone() override;
  IFRIT_HOST virtual std::unique_ptr<Raytracer::CloseHitShader>
  getThreadLocalCopy() override;
  IFRIT_HOST virtual void updateUniformData(int binding, int set,
                                            const void *pData) override;
  IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
  IFRIT_HOST virtual void onStackPushComplete() override;
  IFRIT_HOST virtual void onStackPopComplete() override;
};
} // namespace Ifrit::Engine::SoftRenderer::ShaderVM::Spirv