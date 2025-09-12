#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{
    class VA_Device;

    struct VA_ComputePipelineStateInternal;
    class IFRIT_VKRHI2_API VA_ComputePipelineState : public RHI::RhiComputePipeline
    {
    public:
        VA_ComputePipelineState(const RHI::RhiComputePipelineStateDesc& desc, VA_Device* device);
        virtual ~VA_ComputePipelineState();

        VkPipeline       GetVulkanPipeline() const;
        VkPipelineLayout GetVulkanPipelineLayout() const;

    private:
        VA_ComputePipelineStateInternal* mData = nullptr;
    };

    struct VA_GraphicsPipelineStateInternal;
    class IFRIT_VKRHI2_API VA_GraphicsPipelineState : public RHI::RhiGraphicsPipeline
    {
    public:
        VA_GraphicsPipelineState(const RHI::RhiGraphicsPipelineStateDesc& desc, VA_Device* device);
        virtual ~VA_GraphicsPipelineState();

        VkPipeline       GetVulkanPipeline() const;
        VkPipelineLayout GetVulkanPipelineLayout() const;

    private:
        VA_GraphicsPipelineStateInternal* mData = nullptr;
    };

    struct VA_PipelineStateCacheRegistryInternal;
    class IFRIT_VKRHI2_API VA_PipelineStateCacheRegistry
    {
    public:
        VA_PipelineStateCacheRegistry(VA_Device* device);
        ~VA_PipelineStateCacheRegistry();
        VA_ComputePipelineState*  GetComputePipeline(const RHI::RhiComputePipelineStateDesc& desc);
        VA_GraphicsPipelineState* GetGraphicsPipeline(const RHI::RhiGraphicsPipelineStateDesc& desc);

    private:
        void CreateComputePipeline(const RHI::RhiComputePipelineStateDesc& desc);
        void CreateGraphicsPipeline(const RHI::RhiGraphicsPipelineStateDesc& desc);

    private:
        VA_PipelineStateCacheRegistryInternal* mInternal = nullptr;
    };

} // namespace Ifrit::RHI::VulkanRHI2