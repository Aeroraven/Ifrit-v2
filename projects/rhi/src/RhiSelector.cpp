
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