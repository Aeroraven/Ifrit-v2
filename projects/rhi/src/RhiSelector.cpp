#include "ifrit/rhi/platform/RhiSelector.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Backend.h"
namespace Ifrit::GraphicsBackend::VulkanGraphics {
extern IFRIT_APIDECL_IMPORT void
getRhiBackendBuilder_Vulkan(std::unique_ptr<Rhi::RhiBackendFactory> &ptr);
} // namespace Ifrit::GraphicsBackend::VulkanGraphics

namespace Ifrit::GraphicsBackend::Rhi {
IFRIT_APIDECL std::unique_ptr<RhiBackend>
RhiSelector::createBackend(RhiBackendType type,
                           const RhiInitializeArguments &args) {
  std::unique_ptr<RhiBackendFactory> factory;
  if (type == RhiBackendType::Vulkan) {
    VulkanGraphics::getRhiBackendBuilder_Vulkan(factory);
    return factory->createBackend(args);
  }
  printf("RhiSelector: Backend not found\n");
  return nullptr;
}
} // namespace Ifrit::GraphicsBackend::Rhi