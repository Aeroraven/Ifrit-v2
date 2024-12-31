
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

#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/TypeDescriptor.h"
#include "ifrit/softgraphics/engine/base/VertexBuffer.h"
#include "ifrit/softgraphics/engine/base/VertexShaderResult.h"


namespace Ifrit::GraphicsBackend::SoftGraphics {

struct VaryingDescriptorContext {
  std::vector<TypeDescriptor> varyingDescriptors;
};

class IFRIT_APIDECL VaryingDescriptor {
protected:
  VaryingDescriptorContext *context;

public:
  VaryingDescriptor();
  VaryingDescriptor(const VaryingDescriptor &x) = delete;
  VaryingDescriptor(VaryingDescriptor &&x) IFRIT_NOTHROW;
  ~VaryingDescriptor();
  void
  setVaryingDescriptors(const std::vector<TypeDescriptor> &varyingDescriptors);
  void applyVaryingDescriptors(VertexShaderResult *varyingBuffer);

  /* Inline */
  inline uint32_t getVaryingCounts() const {
    using namespace Ifrit::Common::Utility;
    return size_cast<uint32_t>(context->varyingDescriptors.size());
  }
  inline TypeDescriptor getVaryingDescriptor(int index) const {
    return context->varyingDescriptors[index];
  }

  /* DLL Compat */
  void setVaryingDescriptorsCompatible(const TypeDescriptor *varyingDescriptors,
                                       int num);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics