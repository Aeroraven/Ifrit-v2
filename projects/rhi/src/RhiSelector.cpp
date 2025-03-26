
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
namespace Ifrit::Graphics::VulkanGraphics
{
    extern IFRIT_APIDECL_IMPORT void GetRhiBackendBuilder_Vulkan(Uref<Rhi::RhiBackendFactory>& ptr);
} // namespace Ifrit::Graphics::VulkanGraphics

namespace Ifrit::Graphics::Rhi
{
    IFRIT_APIDECL Uref<RhiBackend> RhiSelector::CreateBackend(RhiBackendType type,
        const RhiInitializeArguments&                                        args)
    {
        Uref<RhiBackendFactory> factory;
        if (type == RhiBackendType::Vulkan)
        {
            VulkanGraphics::GetRhiBackendBuilder_Vulkan(factory);
            return factory->CreateBackend(args);
        }
        printf("RhiSelector: Backend not found\n");
        return nullptr;
    }
} // namespace Ifrit::Graphics::Rhi