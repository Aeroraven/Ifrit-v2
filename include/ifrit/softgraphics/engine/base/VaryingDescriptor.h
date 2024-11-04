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