#include "ifrit/rhi/common/RhiInterface.h"
#include "ifrit/rhi/common/RhiCommandList.h"
#include "ifrit/rhi/common/RhiShaderResource.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/algo/StlStringUtils.h"
namespace Ifrit::RHI
{

    static Owner<RhiBackend> gBackend = nullptr;

    // ===== Backend =====
    struct RhiBackendInternal
    {
        Owner<RhiShaderRegistry> mShaderRegistry;
    };
    IFRIT_RHI_API RhiBackend::RhiBackend()
    {
        mInternal                  = new RhiBackendInternal();
        mInternal->mShaderRegistry = MakeOwner<RhiShaderRegistry>();
    }
    IFRIT_RHI_API RhiBackend::~RhiBackend()
    {
        delete mInternal;
        mInternal = nullptr;
    }

    IFRIT_RHI_API void RhiBackend::InitRenderResources()
    {
        auto executor = GetCommandListExecutor();
        executor->Init();
    }

    IFRIT_RHI_API void RhiBackend::Unload()
    {
        UnloadCommandListExecutor();
        mInternal->mShaderRegistry->RequestShaderRegistryUnload();
        mInternal->mShaderRegistry = nullptr;
    }

    IFRIT_RHI_API RhiShaderRef RhiBackend::CreateShader(const RhiShaderCreateDesc& desc)
    {
        return mInternal->mShaderRegistry->RegisterShader(desc);
    }
    IFRIT_RHI_API RhiShaderRef RhiBackend::GetShader(const String& name)
    {
        return mInternal->mShaderRegistry->GetShader(name);
    }
    IFRIT_RHI_API RhiShaderVariantDesc RhiBackend::GetShaderVariant(const String& name, const Vec<String>& keys)
    {

        auto shader = GetShader(name);
        if (shader == nullptr)
        {
            IF_LOG_WARNING("RhiBackend", "Shader not found: {}", name);
            return {};
        }
        auto variant = shader->GetVariant(keys);
        if (variant == nullptr)
        {
            auto joinedKeys = JoinString(keys, ", ");
            IF_LOG_WARNING("RhiBackend", "Shader variant not found: {} (keys: {})", name, joinedKeys);
            return {};
        }
        return { shader, variant };
    }

    // ===== Global Functions =====

    IFRIT_RHI_API RhiBackend* GetRhiBackend() { return gBackend.get(); }
    IFRIT_RHI_API void        SetRhiBackend(Owner<RhiBackend> backend)
    {
        if (backend.get() != nullptr && gBackend.get() == backend.get())
            return;
        if (gBackend)
            gBackend->Unload();
        gBackend = std::move(backend);
    }
} // namespace Ifrit::RHI