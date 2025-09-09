#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"

namespace Ifrit::RHI::VulkanRHI2
{

    class VA_Device;
    class VA_StagingBufferManager;

    struct VA_BackendInternal;
    class IFRIT_VKRHI2_API VA_Backend : public RhiBackend
    {
    public:
        VA_Backend();
        ~VA_Backend();

        // RhiBackend Overrides
        void                      Init(const RhiInitializeArguments& args) override final;
        void                      Finalize() override final;

        IRhiCommandContext*       GetImmediateContext() override final;
        Owner<IRhiCommandContext> GetUploadContext() override final;
        Owner<IRhiCommandContext> GetCommandContext(ERhiCommandListPipelineType type) override final;
        RhiDynamicUtils*          GetDynamicUtils() override final;

        void                      BeginFrame() override final;
        void                      EndFrame() override final;

        // Resource Creation
        RhiTextureRef             CreateTexture(const RhiTextureDesc& desc) override final;
        RhiBufferRef              CreateBuffer(const RhiBufferDesc& desc) override final;
        RhiUAVRef                 CreateUAV(RhiTexture* texture, RhiImageSubResource subResource) override final;
        RhiUAVRef                 CreateUAV(RhiBuffer* buffer) override final;
        RhiSRVRef                 CreateSRV(RhiTexture* texture, RhiImageSubResource subResource) override final;
        RhiSRVRef                 CreateSRV(RhiBuffer* buffer) override final;

        RhiComputePipeline* Experimental_GetComputePipeline(const RhiComputePipelineStateDesc& desc) override final;

        // VA_Backend specific
        VA_Device*          GetDevice() const;
        VA_StagingBufferManager* GetStagingBufferManager() const;

    private:
        VA_BackendInternal* mInternal;
    };

} // namespace Ifrit::RHI::VulkanRHI2