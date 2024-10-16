#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/TypeDescriptor.h"
#include "engine/base/VaryingStore.h"
#include "math/simd/SimdVectors.h"

namespace Ifrit::Engine::SoftRenderer {
struct VertexShaderResultContext {
  std::vector<ifloat4> position;
  std::vector<std::vector<Ifrit::Math::SIMD::vfloat4>> varyings;
  std::vector<TypeDescriptor> varyingDescriptors;
};
class IFRIT_APIDECL VertexShaderResult {
private:
  VertexShaderResultContext *context;
  uint32_t vertexCount;
  uint32_t rawCounter = 0;

public:
  VertexShaderResult(uint32_t vertexCount, uint32_t varyingCount);
  ~VertexShaderResult();
  ifloat4 *getPositionBuffer();
  void initializeVaryingBufferFromShader(const TypeDescriptor &typeDescriptor,
                                         int id);
  void setVertexCount(const uint32_t vertexCount);

  /* Inline */
  TypeDescriptor getVaryingDescriptor(int id) const {
    return context->varyingDescriptors[id];
  }
  void allocateVaryings(int count) {
    context->varyings.resize(count * 2);
    context->varyingDescriptors.resize(count * 2);
  }
  inline void initializeVaryingBuffer(int id) {
    context->varyings[id].resize(vertexCount);
  }
  inline Ifrit::Math::SIMD::vfloat4 *getVaryingBuffer(int id) {
    return context->varyings[id].data();
  }
};
} // namespace Ifrit::Engine::SoftRenderer