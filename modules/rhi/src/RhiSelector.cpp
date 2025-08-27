#include "ifrit/rhi/platform/RhiSelector.h"
//#include "ifrit/vkrhi/engine/vkrenderer/Backend.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::RHI::VulkanAdapter
{
    //extern IFRIT_APIDECL_IMPORT void GetRhiBackendBuilder_Vulkan(Owner<RHI::RhiBackendFactory>& ptr);
} // namespace Ifrit::RHI::VulkanAdapter

namespace Ifrit::RHI
{
    IFRIT_APIDECL Owner<RhiBackend> RhiSelector::CreateBackend(RhiBackendType type, const RhiInitializeArguments& args)
    {
        Owner<RhiBackendFactory> factory;
        if (type == RhiBackendType::Vulkan)
        {
            //VulkanAdapter::GetRhiBackendBuilder_Vulkan(factory);
            //return factory->CreateBackend(args);
        }
        IF_LOG_CRITICAL("RhiSelector", "Unsupported RHI backend type: {}", static_cast<int>(type));
        return nullptr;
    }
} // namespace Ifrit::RHI