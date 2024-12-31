
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


#include "ifrit/softgraphics/engine/base/VaryingDescriptor.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
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
} // namespace Ifrit::GraphicsBackend::SoftGraphics