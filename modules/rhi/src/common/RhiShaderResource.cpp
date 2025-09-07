#include "ifrit/rhi/common/RhiShaderResource.h"
#include "ifrit/core/console/ConsoleObject.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/rhi/common/RhiInterface.h"
#include "ifrit/rhi/common/RhiDynamicUtils.h"

namespace Ifrit::RHI
{
    static TConsoleVariable<u32> cvRHIMaxShaderMultiCompiles(
        "cv.RHI.MaxMultiCompiles", 12, "Maximum number of multi compile directives allowed in a shader", CVF_ReadOnly);

    IFRIT_RHI_API u32 GetMaxMultiCompileDirectives() { return cvRHIMaxShaderMultiCompiles.GetValue(); }

    // ===== Shader Registry =====
    struct RhiShaderRegistryInternal : public NonCopyableStruct
    {
        HashMap<String, RhiShaderRef> mRegisteredShaders;
    };

    RhiShaderRegistry::RhiShaderRegistry() { mInternal = new RhiShaderRegistryInternal(); }

    RhiShaderRegistry::~RhiShaderRegistry() { delete mInternal; }

    IFRIT_APIDECL RhiShaderRef RhiShaderRegistry::GetShader(const String& name)
    {
        auto it = mInternal->mRegisteredShaders.find(name);
        if (it != mInternal->mRegisteredShaders.end())
        {
            return it->second;
        }
        IF_LOG_WARNING("RhiShaderRegistry", "Shader not found: {}", name);
        return nullptr;
    }

    IFRIT_APIDECL RhiShaderRef RhiShaderRegistry::RegisterShader(const RhiShaderCreateDesc& desc)
    {
        auto backend                              = GetRhiBackend();
        auto dynUtils                             = backend->GetDynamicUtils();
        auto shader                               = dynUtils->CreateShader_RhiInternal(desc);
        mInternal->mRegisteredShaders[desc.mName] = shader;
        return shader;
    }

} // namespace Ifrit::RHI