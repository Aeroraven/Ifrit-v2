
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

#include "ifrit/vkgraphics/engine/vkrenderer/StagedMemoryResource.h"
#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Graphics::VulkanGraphics
{

    IFRIT_APIDECL StagedSingleBuffer::StagedSingleBuffer(EngineContext* ctx, SingleBuffer* buffer)
    {
        m_context = ctx;
        m_buffer  = buffer;
        BufferCreateInfo stagingCI{};
        stagingCI.size        = buffer->GetSize();
        stagingCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingCI.hostVisible = true;
        auto stagePtr         = new SingleBuffer(ctx, stagingCI);
        m_stagingBuffer       = Rhi::MakeRhiCountRef<Rhi::RhiBuffer>(stagePtr);
    }
    IFRIT_APIDECL StagedSingleBuffer::StagedSingleBuffer(EngineContext* ctx, const BufferCreateInfo& ci)
        : m_context(ctx)
    {

        using namespace Ifrit::Common::Utility;
        BufferCreateInfo ci2 = ci;
        ci2.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        auto stagePtr  = new SingleBuffer(ctx, ci2);
        m_bufferUnique = Rhi::MakeRhiCountRef<Rhi::RhiBuffer>(stagePtr);
        m_buffer       = CheckedCast<SingleBuffer>(m_bufferUnique.get());

        BufferCreateInfo stagingCI = ci;
        stagingCI.usage            = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingCI.hostVisible      = true;
        auto stagePtr2             = new SingleBuffer(ctx, stagingCI);
        m_stagingBuffer            = Rhi::MakeRhiCountRef<Rhi::RhiBuffer>(stagePtr2);
    }

    IFRIT_APIDECL void StagedSingleBuffer::CmdCopyToDevice(const Rhi::RhiCommandList* cmd, const void* data, u32 size,
        u32 localOffset)
    {
        m_stagingBuffer->MapMemory();
        m_stagingBuffer->WriteBuffer((void*)data, size, 0);
        m_stagingBuffer->FlushBuffer();
        m_stagingBuffer->UnmapMemory();
        cmd->CopyBuffer(m_stagingBuffer.get(), m_buffer, size, 0, localOffset);
    }

    IFRIT_APIDECL StagedSingleImage::StagedSingleImage(EngineContext* ctx, SingleDeviceImage* image)
        : m_context(ctx), m_image(image)
    {
        BufferCreateInfo stagingCI{};
        stagingCI.size        = image->GetSize();
        stagingCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingCI.hostVisible = true;
        auto stagePtr         = new SingleBuffer(ctx, stagingCI);
        m_stagingBuffer       = Rhi::MakeRhiCountRef<Rhi::RhiBuffer>(stagePtr);
    }

    IFRIT_APIDECL StagedSingleImage::StagedSingleImage(EngineContext* ctx, const ImageCreateInfo& ci)
        : m_context(ctx)
    {
        using namespace Ifrit::Common::Utility;
        ImageCreateInfo ci2 = ci;
        ci2.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        auto stagePtr = new SingleDeviceImage(ctx, ci2);
        m_imageUnique = Rhi::MakeRhiCountRef<Rhi::RhiTexture>(stagePtr);
        m_image       = CheckedCast<SingleDeviceImage>(m_imageUnique.get());

        BufferCreateInfo stagingCI{};
        stagingCI.size        = m_image->GetSize();
        stagingCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingCI.hostVisible = true;
        auto stagePtr2        = new SingleBuffer(ctx, stagingCI);
        m_stagingBuffer       = Rhi::MakeRhiCountRef<Rhi::RhiBuffer>(stagePtr2);
    }

    IFRIT_APIDECL void StagedSingleImage::CmdCopyToDevice(CommandBuffer* cmd, const void* data, VkImageLayout srcLayout,
        VkImageLayout dstLayout, VkPipelineStageFlags dstStage,
        VkAccessFlags dstAccess)
    {
        m_stagingBuffer->MapMemory();
        m_stagingBuffer->WriteBuffer((void*)data, m_image->GetSize(), 0);
        m_stagingBuffer->FlushBuffer();
        m_stagingBuffer->UnmapMemory();
        PipelineBarrier      barrier(m_context, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0);

        VkImageMemoryBarrier imageBarrier{};
        imageBarrier.sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        imageBarrier.oldLayout                   = srcLayout;
        imageBarrier.newLayout                   = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imageBarrier.srcQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.dstQueueFamilyIndex         = VK_QUEUE_FAMILY_IGNORED;
        imageBarrier.image                       = m_image->GetImage();
        imageBarrier.subresourceRange.aspectMask = m_image->GetAspect();
        imageBarrier.subresourceRange.levelCount = 1;
        imageBarrier.subresourceRange.layerCount = 1;
        imageBarrier.srcAccessMask               = 0;
        imageBarrier.dstAccessMask               = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.addImageMemoryBarrier(imageBarrier);

        cmd->AddPipelineBarrier(barrier);
        cmd->CopyBufferToImageAllInternal(m_stagingBuffer.get(), m_image->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            m_image->GetWidth(), m_image->GetHeight(), m_image->GetDepth());

        imageBarrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        imageBarrier.newLayout     = dstLayout;
        imageBarrier.srcAccessMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        imageBarrier.dstAccessMask = dstAccess;
        PipelineBarrier barrier2(m_context, VK_ACCESS_TRANSFER_WRITE_BIT, dstStage, 0);
        barrier2.addImageMemoryBarrier(imageBarrier);
        cmd->AddPipelineBarrier(barrier2);
    }

} // namespace Ifrit::Graphics::VulkanGraphics