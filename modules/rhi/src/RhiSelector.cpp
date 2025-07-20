
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
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::RHI::VulkanAdapter
{
    extern IFRIT_APIDECL_IMPORT void GetRhiBackendBuilder_Vulkan(Owner<RHI::RhiBackendFactory>& ptr);
} // namespace Ifrit::RHI::VulkanAdapter

namespace Ifrit::RHI
{
    IFRIT_APIDECL Owner<RhiBackend> RhiSelector::CreateBackend(RhiBackendType type, const RhiInitializeArguments& args)
    {
        Owner<RhiBackendFactory> factory;
        if (type == RhiBackendType::Vulkan)
        {
            VulkanAdapter::GetRhiBackendBuilder_Vulkan(factory);
            return factory->CreateBackend(args);
        }
        IF_LOG_CRITICAL("RhiSelector", "Unsupported RHI backend type: {}", static_cast<int>(type));
        return nullptr;
    }
} // namespace Ifrit::RHI