#include "ifrit/rhi/platform/RhiSelector.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Backend.h"
namespace Ifrit::GraphicsBackend::Rhi {
IFRIT_APIDECL std::unique_ptr<RhiBackend>
RhiSelector::createBackend(RhiBackendType type,
                           const RhiInitializeArguments &args) {
  if (type == RhiBackendType::Vulkan) {
    VulkanGraphics::RhiVulkanBackendBuilder builder;
    return builder.createBackend(args);
  }
  printf("RhiSelector: Backend not found\n");
  return nullptr;
}
} // namespace Ifrit::GraphicsBackend::Rhi