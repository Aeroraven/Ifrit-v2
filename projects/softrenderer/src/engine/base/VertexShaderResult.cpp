#include "engine/base/VertexShaderResult.h"

namespace Ifrit::Engine::SoftRenderer {

IFRIT_APIDECL VertexShaderResult::VertexShaderResult(uint32_t vertexCount,
                                                     uint32_t varyingCount) {
  this->context = new std::remove_pointer_t<decltype(this->context)>();
  this->vertexCount = vertexCount;
  this->context->varyings.resize(varyingCount);
}
IFRIT_APIDECL VertexShaderResult::~VertexShaderResult() {
  delete this->context;
}

IFRIT_APIDECL ifloat4 *VertexShaderResult::getPositionBuffer() {
  return context->position.data();
}
IFRIT_APIDECL void VertexShaderResult::initializeVaryingBufferFromShader(
    const TypeDescriptor &typeDescriptor, int id) {
  this->context->varyings[id].resize(vertexCount * typeDescriptor.size);
  this->context->varyingDescriptors[id] = typeDescriptor;
}
IFRIT_APIDECL void VertexShaderResult::setVertexCount(const uint32_t vcnt) {
  this->vertexCount = vcnt;
  for (auto &varying : context->varyings) {
    varying.resize(vertexCount * sizeof(ifloat4));
    context->varyingDescriptors.resize(context->varyings.size());
  }
  context->position.resize(vertexCount);
}
} // namespace Ifrit::Engine::SoftRenderer