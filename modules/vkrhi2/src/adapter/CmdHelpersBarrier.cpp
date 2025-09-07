#include "ifrit.internal/vkrhi2/adapter/CmdHelpersBarrier.h"
#include "ifrit/vkrhi2/adapter/MemoryResource.h"

namespace Ifrit::RHI::VulkanRHI2
{
    VkAccessFlags2 GetAccessFlagsFromImageLayout(VkImageLayout layout)
    {
        switch (layout)
        {
            case VK_IMAGE_LAYOUT_UNDEFINED:
            case VK_IMAGE_LAYOUT_PREINITIALIZED:
                return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

            case VK_IMAGE_LAYOUT_GENERAL:
                return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;

            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return VK_ACCESS_2_SHADER_READ_BIT;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return VK_ACCESS_2_TRANSFER_READ_BIT;

            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return VK_ACCESS_2_TRANSFER_WRITE_BIT;

            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                return VK_ACCESS_2_MEMORY_READ_BIT;

            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported image layout");
        }
        return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }

    VkPipelineStageFlags2 GetAccessStagesFromImageLayout(VkImageLayout layout)
    {
        switch (layout)
        {
            case VK_IMAGE_LAYOUT_UNDEFINED:
            case VK_IMAGE_LAYOUT_PREINITIALIZED:
                return VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;

            case VK_IMAGE_LAYOUT_GENERAL:
                return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                return VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;

            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
                return VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
                    | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;

            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                    | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_NV | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                    | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT
                    | VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT;

            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                return VK_PIPELINE_STAGE_2_TRANSFER_BIT;

            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return VK_PIPELINE_STAGE_2_TRANSFER_BIT;

            case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
                return VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;

            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported image layout");
        }
        return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    VkAccessFlags2 InternalTranslateBufferAccessFlags(ERhiResourceState access)
    {
        switch (access)
        {
            case ERhiResourceState::UnorderedAccess:
                return VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            case ERhiResourceState::UnorderedAccess_Read:
                return VK_ACCESS_2_SHADER_READ_BIT;
            case ERhiResourceState::UnorderedAccess_Write:
                return VK_ACCESS_2_SHADER_WRITE_BIT;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported buffer access flag");
        }
        return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }
    VkPipelineStageFlags2 InternalTranslateBufferPipelineStages(ERhiResourceState access, ERhiPipelineType pipeline)
    {
        VkPipelineStageFlags2 stage = 0;
        switch (pipeline)
        {
            case ERhiPipelineType::Graphics:
                stage |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
                break;
            case ERhiPipelineType::Compute:
                stage |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                break;
            case ERhiPipelineType::Transfer:
                stage |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                break;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported pipeline type");
        }
        switch (access)
        {
            case ERhiResourceState::UnorderedAccess:
            case ERhiResourceState::UnorderedAccess_Read:
            case ERhiResourceState::UnorderedAccess_Write:
                stage |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                break;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported buffer access flag");
        }
        return stage;
    }

    VkAccessFlags2 InternalTranslateImageAccessFlags(ERhiResourceState access)
    {
        switch (access)
        {
            case ERhiResourceState::ColorRT:
                return VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
            case ERhiResourceState::DepthStencilRT:
                return VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            case ERhiResourceState::ShaderRead:
                return VK_ACCESS_2_SHADER_READ_BIT;
            case ERhiResourceState::UnorderedAccess:
                return VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
            case ERhiResourceState::UnorderedAccess_Read:
                return VK_ACCESS_2_SHADER_READ_BIT;
            case ERhiResourceState::UnorderedAccess_Write:
                return VK_ACCESS_2_SHADER_WRITE_BIT;
            case ERhiResourceState::CopySrc:
                return VK_ACCESS_2_TRANSFER_READ_BIT;
            case ERhiResourceState::CopyDst:
                return VK_ACCESS_2_TRANSFER_WRITE_BIT;
            case ERhiResourceState::Present:
                return VK_ACCESS_2_MEMORY_READ_BIT;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported image access flag");
        }
        return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }

    VkPipelineStageFlags2 InternalTranslateImagePipelineStages(ERhiResourceState access, ERhiPipelineType pipeline)
    {
        VkPipelineStageFlags2 stage = 0;
        switch (pipeline)
        {
            case ERhiPipelineType::Graphics:
                stage |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
                break;
            case ERhiPipelineType::Compute:
                stage |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                break;
            case ERhiPipelineType::Transfer:
                stage |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                break;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported pipeline type");
        }
        switch (access)
        {
            case ERhiResourceState::ColorRT:
            case ERhiResourceState::DepthStencilRT:
            case ERhiResourceState::ShaderRead:
            case ERhiResourceState::UnorderedAccess:
            case ERhiResourceState::UnorderedAccess_Read:
            case ERhiResourceState::UnorderedAccess_Write:
                stage |= VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                break;
            case ERhiResourceState::CopySrc:
            case ERhiResourceState::CopyDst:
                stage |= VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                break;
            case ERhiResourceState::Present:
                stage |= VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
                break;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported image access flag");
        }
        return stage;
    }
    VkImageLayout InternalTranslateImageLayout(ERhiResourceState access)
    {
        switch (access)
        {
            case ERhiResourceState::ColorRT:
                return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            case ERhiResourceState::DepthStencilRT:
                return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            case ERhiResourceState::ShaderRead:
                return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            case ERhiResourceState::UnorderedAccess:
            case ERhiResourceState::UnorderedAccess_Read:
            case ERhiResourceState::UnorderedAccess_Write:
                return VK_IMAGE_LAYOUT_GENERAL;
            case ERhiResourceState::CopySrc:
                return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            case ERhiResourceState::CopyDst:
                return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            case ERhiResourceState::Present:
                return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported image access flag");
        }
        return VK_IMAGE_LAYOUT_GENERAL;
    }

    // ===== VA_PipelineBarriers =====

    void VA_PipelineBarriers::AddImageTransition(
        VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageSubresourceRange subResource)
    {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.oldLayout           = oldLayout;
        barrier.newLayout           = newLayout;
        barrier.srcAccessMask       = GetAccessFlagsFromImageLayout(oldLayout);
        barrier.dstAccessMask       = GetAccessFlagsFromImageLayout(newLayout);
        barrier.srcStageMask        = GetAccessStagesFromImageLayout(oldLayout);
        barrier.dstStageMask        = GetAccessStagesFromImageLayout(newLayout);
        barrier.image               = image;
        barrier.subresourceRange    = subResource;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

        mImageBarriers.push_back(barrier);
    }

    void VA_PipelineBarriers::AddMemoryBarrier(VkPipelineStageFlags2 srcStageMask, VkPipelineStageFlags2 dstStageMask,
        VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask, VkDependencyFlags dependencyFlags)
    {
        VkMemoryBarrier2 barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask  = srcStageMask;
        barrier.dstStageMask  = dstStageMask;
        barrier.srcAccessMask = srcAccessMask;
        barrier.dstAccessMask = dstAccessMask;

        mMemoryBarriers.push_back(barrier);
    }

    u32 VA_PipelineBarriers::TranslateQueueFamilyIndex(ERhiPipelineType pipeline)
    {
        if (!mQueueInfoSpecified)
        {
            IF_LOG_ASSERTION("VAHelper_Barrier", mQueueInfoSpecified,
                "Queue family info must be specified for cross-queue barriers");
            return VK_QUEUE_FAMILY_IGNORED;
        }
        switch (pipeline)
        {
            case ERhiPipelineType::Graphics:
                return mQueueInfo.mGraphics;
            case ERhiPipelineType::Compute:
                return mQueueInfo.mAsyncCompute;
            case ERhiPipelineType::Transfer:
                return mQueueInfo.mTransfer;
            default:
                IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported pipeline type");
        }
        return VK_QUEUE_FAMILY_IGNORED;
    }

    VkBufferMemoryBarrier2 VA_PipelineBarriers::TranslateBufferMemBarrierFromRhi(
        const RHI::RhiResourceTransitionDesc& desc, bool isBeginStage, ERhiPipelineType srcPipeline,
        ERhiPipelineType dstPipeline)
    {
        IF_LOG_ASSERTION("VAHelper_Barrier", desc.mBuffer != nullptr, "Buffer transition with null buffer");
        IF_LOG_ASSERTION("VAHelper_Barrier", desc.mType == ERhiResourceType::Buffer,
            "Buffer transition with non-buffer resource type");

        VkBufferMemoryBarrier2 ret{};
        ret.sType  = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        ret.buffer = reinterpret_cast<VkBuffer>(CheckedCast<VA_Buffer>(desc.mBuffer)->GetRawHandle());
        ret.size   = VK_WHOLE_SIZE;
        ret.offset = 0;

        bool requireSeparateCmdBuf = (srcPipeline != dstPipeline);
        if (srcPipeline == dstPipeline)
        {
            ret.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ret.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        }
        else
        {
            ret.srcQueueFamilyIndex = TranslateQueueFamilyIndex(srcPipeline);
            ret.dstQueueFamilyIndex = TranslateQueueFamilyIndex(dstPipeline);
        }
        ret.srcAccessMask = InternalTranslateBufferAccessFlags(desc.mSrcState);
        ret.dstAccessMask = InternalTranslateBufferAccessFlags(desc.mDstState);
        ret.srcStageMask  = InternalTranslateBufferPipelineStages(desc.mSrcState, srcPipeline);
        ret.dstStageMask  = InternalTranslateBufferPipelineStages(desc.mDstState, dstPipeline);

        if (requireSeparateCmdBuf)
        {
            if (!isBeginStage)
            {
                ret.srcAccessMask = 0;
                ret.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
            else
            {
                ret.dstAccessMask = 0;
                ret.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
        }
        return ret;
    }

    VkImageMemoryBarrier2 VA_PipelineBarriers::TranslateImageMemBarrierFromRhi(
        const RHI::RhiResourceTransitionDesc& desc, bool isBeginStage, ERhiPipelineType srcPipeline,
        ERhiPipelineType dstPipeline)
    {
        IF_LOG_ASSERTION("VAHelper_Barrier", desc.mTexture != nullptr, "Image transition with null image");
        IF_LOG_ASSERTION("VAHelper_Barrier", desc.mType == ERhiResourceType::Texture,
            "Image transition with non-image resource type");
        VkImageMemoryBarrier2 ret{};
        auto                  tex = CheckedCast<VA_Texture>(desc.mTexture);

        ret.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        ret.image = reinterpret_cast<VkImage>(tex->GetRawHandle());
        if (tex->IsDepthTexture())
        {
            ret.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (tex->GetVkFormat() == VK_FORMAT_D32_SFLOAT_S8_UINT || tex->GetVkFormat() == VK_FORMAT_D24_UNORM_S8_UINT)
            {
                ret.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }
        else
        {
            ret.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        ret.subresourceRange.baseMipLevel   = desc.mSubResource.mipLevel;
        ret.subresourceRange.levelCount     = desc.mSubResource.mipCount;
        ret.subresourceRange.baseArrayLayer = desc.mSubResource.arrayLayer;
        ret.subresourceRange.layerCount     = desc.mSubResource.layerCount;
        bool requireSeparateCmdBuf          = (srcPipeline != dstPipeline);
        if (srcPipeline == dstPipeline)
        {
            ret.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            ret.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        }
        else
        {
            ret.srcQueueFamilyIndex = TranslateQueueFamilyIndex(srcPipeline);
            ret.dstQueueFamilyIndex = TranslateQueueFamilyIndex(dstPipeline);
        }
        ret.srcAccessMask = InternalTranslateImageAccessFlags(desc.mSrcState);
        ret.dstAccessMask = InternalTranslateImageAccessFlags(desc.mDstState);
        ret.srcStageMask  = InternalTranslateImagePipelineStages(desc.mSrcState, srcPipeline);
        ret.dstStageMask  = InternalTranslateImagePipelineStages(desc.mDstState, dstPipeline);
        ret.oldLayout     = InternalTranslateImageLayout(desc.mSrcState);
        ret.newLayout     = InternalTranslateImageLayout(desc.mDstState);
        if (requireSeparateCmdBuf)
        {
            if (!isBeginStage)
            {
                ret.srcAccessMask = 0;
                ret.srcStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
            else
            {
                ret.dstAccessMask = 0;
                ret.dstStageMask  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            }
        }
        return ret;
    }

    void VA_PipelineBarriers::TranslateFromRhiBarriers(const RHI::RhiTransition& transition, bool isBeginStage)
    {
        if (!isBeginStage && transition.mPipelineDst == transition.mPipelineSrc)
            return;
        bool requireSplitCmdBuf = (transition.mPipelineDst != transition.mPipelineSrc);
        for (const auto& barrier : transition.mTransitions)
        {
            switch (barrier.mType)
            {
                case ERhiResourceType::Buffer:
                {
                    auto vkBarrier = TranslateBufferMemBarrierFromRhi(
                        barrier, isBeginStage, transition.mPipelineSrc, transition.mPipelineDst);
                    mBufferBarriers.push_back(vkBarrier);
                }
                break;
                case ERhiResourceType::Texture:
                {
                    auto vkBarrier = TranslateImageMemBarrierFromRhi(
                        barrier, isBeginStage, transition.mPipelineSrc, transition.mPipelineDst);
                    mImageBarriers.push_back(vkBarrier);
                }
                break;
                default:
                    IF_LOG_CRITICAL("VAHelper_Barrier", "Unsupported resource type in transition");
            }
        }
    }

    void VA_PipelineBarriers::ExecuteNative(VkCommandBuffer cmd)
    {
        if (mMemoryBarriers.empty() && mBufferBarriers.empty() && mImageBarriers.empty())
            return;

        VkDependencyInfo depInfo{};
        depInfo.sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.memoryBarrierCount       = static_cast<u32>(mMemoryBarriers.size());
        depInfo.pMemoryBarriers          = mMemoryBarriers.data();
        depInfo.bufferMemoryBarrierCount = static_cast<u32>(mBufferBarriers.size());
        depInfo.pBufferMemoryBarriers    = mBufferBarriers.data();
        depInfo.imageMemoryBarrierCount  = static_cast<u32>(mImageBarriers.size());
        depInfo.pImageMemoryBarriers     = mImageBarriers.data();
        depInfo.dependencyFlags          = 0;

        vkCmdPipelineBarrier2(cmd, &depInfo);

        mMemoryBarriers.clear();
        mBufferBarriers.clear();
        mImageBarriers.clear();
    }

} // namespace Ifrit::RHI::VulkanRHI2