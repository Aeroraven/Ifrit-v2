#pragma once

#include "ifrit/common/util/ApiConv.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <memory>

namespace Ifrit::GraphicsBackend::Rhi {
enum class RhiBackendType { Vulkan, DirectX, OpenGL, Software };
class IFRIT_APIDECL RhiSelector {
public:
  std::unique_ptr<RhiBackend> createBackend(RhiBackendType,
                                            const RhiInitializeArguments &args);
};
} // namespace Ifrit::GraphicsBackend::Rhi