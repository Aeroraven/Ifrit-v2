#include "ifrit/softgraphics/engine/base/VaryingDescriptor.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics {
IFRIT_APIDECL
VaryingDescriptor::VaryingDescriptor(VaryingDescriptor &&x) IFRIT_NOTHROW {
  this->context = x.context;
  x.context = nullptr;
}
IFRIT_APIDECL void VaryingDescriptor::setVaryingDescriptors(
    const std::vector<TypeDescriptor> &varyingDescriptors) {
  this->context->varyingDescriptors = varyingDescriptors;
}
IFRIT_APIDECL void
VaryingDescriptor::applyVaryingDescriptors(VertexShaderResult *varyingBuffer) {
  for (int i = 0; i < context->varyingDescriptors.size(); i++) {
    varyingBuffer->initializeVaryingBufferFromShader(
        context->varyingDescriptors[i], i);
  }
}
IFRIT_APIDECL VaryingDescriptor::VaryingDescriptor() {
  this->context = new std::remove_pointer_t<decltype(this->context)>();
}
IFRIT_APIDECL VaryingDescriptor::~VaryingDescriptor() {
  if (this->context)
    delete this->context;
}

/* DLL Compatible */
IFRIT_APIDECL void VaryingDescriptor::setVaryingDescriptorsCompatible(
    const TypeDescriptor *varyingDescriptors, int num) {
  this->context->varyingDescriptors = std::vector<TypeDescriptor>(num);
  for (int i = 0; i < num; i++) {
    this->context->varyingDescriptors[i] = varyingDescriptors[i];
  }
}
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics