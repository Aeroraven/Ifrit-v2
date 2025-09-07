#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{

    VkPipelineStageFlags2 GetAccessStagesFromImageLayout(VkImageLayout layout);
    VkAccessFlags2        GetAccessFlagsFromImageLayout(VkImageLayout layout);

    struct VA_PipelineBarriers
    {
        Vec<VkMemoryBarrier2>       mMemoryBarriers;
        Vec<VkBufferMemoryBarrier2> mBufferBarriers;
        Vec<VkImageMemoryBarrier2>  mImageBarriers;
        VA_ActiveQueueFamilyInfo    mQueueInfo;
        bool                        mQueueInfoSpecified = false;

    public:
        inline void SetQueueInfo(const VA_ActiveQueueFamilyInfo& info)
        {
            mQueueInfo          = info;
            mQueueInfoSpecified = true;
        }
        void AddImageTransition(
            VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subResource);
        void                   AddMemoryBarrier(VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask,
                              VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask, VkDependencyFlags dependencyFlags = 0);

        u32                    TranslateQueueFamilyIndex(ERhiPipelineType pipeline);
        VkBufferMemoryBarrier2 TranslateBufferMemBarrierFromRhi(const RHI::RhiResourceTransitionDesc& desc,
            bool isBeginStage, ERhiPipelineType srcPipeline, ERhiPipelineType dstPipeline);
        VkImageMemoryBarrier2  TranslateImageMemBarrierFromRhi(const RHI::RhiResourceTransitionDesc& desc,
             bool isBeginStage, ERhiPipelineType srcPipeline, ERhiPipelineType dstPipeline);
        void                   TranslateFromRhiBarriers(const RHI::RhiTransition& transition, bool isBeginStage);

        void                   ExecuteNative(VkCommandBuffer cmd);
    };

    inline void CmdLegacyGlobalPipelineBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStageMask,
        VkPipelineStageFlags dstStageMask, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask)
    {
        VkMemoryBarrier barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;
        vkCmdPipelineBarrier(cmd, srcStageMask, dstStageMask, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }

} // namespace Ifrit::RHI::VulkanRHI2