
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

#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Binding.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkgraphics/engine/vkrenderer/RenderGraph.h"
#include "ifrit/vkgraphics/utility/Logger.h"

using namespace Ifrit;

namespace Ifrit::Graphics::VulkanGraphics
{
    IFRIT_APIDECL TimelineSemaphore::TimelineSemaphore(EngineContext* ctx) : m_context(ctx)
    {
        VkSemaphoreTypeCreateInfo timelineCI{};
        timelineCI.sType         = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        timelineCI.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineCI.initialValue  = 0;
        VkSemaphoreCreateInfo semaphoreCI{};
        semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semaphoreCI.pNext = &timelineCI;
        vkrVulkanAssert(vkCreateSemaphore(m_context->GetDevice(), &semaphoreCI, nullptr, &m_semaphore),
            "Failed to create timeline semaphore");
    }

    IFRIT_APIDECL TimelineSemaphore::~TimelineSemaphore()
    {
        vkDestroySemaphore(m_context->GetDevice(), m_semaphore, nullptr);
    }

    IFRIT_APIDECL void CommandPool::Init()
    {
        VkCommandPoolCreateInfo poolCI{};
        poolCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolCI.queueFamilyIndex = m_queueFamily;
        poolCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkrVulkanAssert(vkCreateCommandPool(m_context->GetDevice(), &poolCI, nullptr, &m_commandPool),
            "Failed to create command pool");
    }

    IFRIT_APIDECL void CommandPool::ResetCommandPool()
    {
        // Free all command buffers in the pool
        // vkrVulkanAssert(
        //     vkResetCommandPool(m_context->GetDevice(), m_commandPool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT),
        //     "Failed to reset command pool");

        // Make all in-flight command buffers available again
        for (auto& cmdBuf : m_InFlightCommandBuffers)
        {
            auto vkbuf = cmdBuf->GetCommandBuffer();
            vkrVulkanAssert(vkResetCommandBuffer(vkbuf, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT),
                "Failed to reset command buffer");
            m_AvailableCommandBuffers.push_back(std::move(cmdBuf));
        }
        m_InFlightCommandBuffers.clear();
    }

    IFRIT_APIDECL void CommandPool::EnqueueInFlightCommandBuffer(Uref<CommandBuffer>&& cmdBuf)
    {
        m_InFlightCommandBuffers.emplace_back(std::move(cmdBuf));
    }

    IFRIT_APIDECL CommandPool::~CommandPool() { vkDestroyCommandPool(m_context->GetDevice(), m_commandPool, nullptr); }

    IFRIT_APIDECL Ref<CommandBuffer> CommandPool::AllocateCommandBuffer()
    {

        VkCommandBufferAllocateInfo bufferAI{};
        bufferAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        bufferAI.commandPool        = m_commandPool;
        bufferAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        bufferAI.commandBufferCount = 1;

        VkCommandBuffer buffer;
        vkrVulkanAssert(
            vkAllocateCommandBuffers(m_context->GetDevice(), &bufferAI, &buffer), "Failed to allocate command buffer");
        return std::make_shared<CommandBuffer>(m_context, buffer, m_queueFamily);
    }

    IFRIT_APIDECL std::unique_ptr<CommandBuffer> CommandPool::AllocateCommandBufferUnique()
    {
        if (m_AvailableCommandBuffers.empty())
        {
            VkCommandBufferAllocateInfo bufferAI{};
            bufferAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            bufferAI.commandPool        = m_commandPool;
            bufferAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            bufferAI.commandBufferCount = 1;

            VkCommandBuffer buffer;
            vkrVulkanAssert(vkAllocateCommandBuffers(m_context->GetDevice(), &bufferAI, &buffer),
                "Failed to allocate command buffer");

            return std::make_unique<CommandBuffer>(m_context, buffer, m_queueFamily);
        }
        else
        {
            auto cmdBuf = std::move(m_AvailableCommandBuffers.back());
            m_AvailableCommandBuffers.pop_back();
            return cmdBuf;
        }
    }

    // Class: Pipeline Barrier
    IFRIT_APIDECL void PipelineBarrier::addMemoryBarrier(VkMemoryBarrier barrier)
    {
        m_memoryBarriers.push_back(barrier);
    }

    IFRIT_APIDECL void PipelineBarrier::addBufferMemoryBarrier(VkBufferMemoryBarrier barrier)
    {
        m_bufferMemoryBarriers.push_back(barrier);
    }

    IFRIT_APIDECL void PipelineBarrier::addImageMemoryBarrier(VkImageMemoryBarrier barrier)
    {
        m_imageMemoryBarriers.push_back(barrier);
    }

    // Class: CommandBuffer
    struct CommandListContextPrivate
    {
        enum class BoundType
        {
            None,
            Graphics,
            Compute,
        };
        BoundType m_BoundType = BoundType::None;
        union
        {
            Rhi::RhiGraphicsPass* m_GraphicsPass;
            Rhi::RhiComputePass*  m_ComputePass;
        } m_BoundPass;
    };

    IFRIT_APIDECL void CommandBuffer::InitPost() { m_CmdContext = new CommandListContextPrivate(); }
    IFRIT_APIDECL void CommandBuffer::DestroyPost() { delete m_CmdContext; }
    IFRIT_APIDECL void CommandBuffer::BeginRecord()
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkrVulkanAssert(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "Failed to begin command buffer");
    }

    IFRIT_APIDECL void CommandBuffer::EndRecord()
    {
        vkrVulkanAssert(vkEndCommandBuffer(m_commandBuffer), "Failed to end command buffer");
    }

    IFRIT_APIDECL void CommandBuffer::Reset()
    {
        vkrVulkanAssert(vkResetCommandBuffer(m_commandBuffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT),
            "Failed to reset command buffer");
    }

    IFRIT_APIDECL void CommandBuffer::AddPipelineBarrier(const PipelineBarrier& barrier) const
    {
        std::abort();
        vkCmdPipelineBarrier(m_commandBuffer, barrier.m_srcStage, barrier.m_dstStage, barrier.m_dependencyFlags,
            SizeCast<int>(barrier.m_memoryBarriers.size()), barrier.m_memoryBarriers.data(),
            SizeCast<int>(barrier.m_bufferMemoryBarriers.size()), barrier.m_bufferMemoryBarriers.data(),
            SizeCast<int>(barrier.m_imageMemoryBarriers.size()), barrier.m_imageMemoryBarriers.data());
    }

    IFRIT_APIDECL void CommandBuffer::BindGraphicsInternal(Rhi::RhiGraphicsPass* pass)
    {
        m_CmdContext->m_BoundType                = CommandListContextPrivate::BoundType::Graphics;
        m_CmdContext->m_BoundPass.m_GraphicsPass = pass;
    }

    IFRIT_APIDECL void CommandBuffer::BindComputeInternal(Rhi::RhiComputePass* pass)
    {
        m_CmdContext->m_BoundType               = CommandListContextPrivate::BoundType::Compute;
        m_CmdContext->m_BoundPass.m_ComputePass = pass;
    }

    IFRIT_APIDECL void CommandBuffer::SetViewports(const Vec<Rhi::RhiViewport>& viewport) const
    {
        Vec<VkViewport> vps;
        for (int i = 0; i < viewport.size(); i++)
        {
            VkViewport s = { viewport[i].x, viewport[i].y, viewport[i].width, viewport[i].height, viewport[i].minDepth,
                viewport[i].maxDepth };
            vps.push_back(s);
        }
        vkCmdSetViewport(m_commandBuffer, 0, SizeCast<int>(vps.size()), vps.data());
    }

    IFRIT_APIDECL void CommandBuffer::SetScissors(const Vec<Rhi::RhiScissor>& scissor) const
    {
        Vec<VkRect2D> scs;
        for (int i = 0; i < scissor.size(); i++)
        {
            VkRect2D s = { { scissor[i].x, scissor[i].y }, { scissor[i].width, scissor[i].height } };
            scs.push_back(s);
        }
        vkCmdSetScissor(m_commandBuffer, 0, SizeCast<int>(scs.size()), scs.data());
    }

    IFRIT_APIDECL void CommandBuffer::Draw(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const
    {
        vkCmdDraw(m_commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
    }

    IFRIT_APIDECL void CommandBuffer::DrawIndexed(
        u32 indexCount, u32 instanceCount, u32 firstIndex, int32_t vertexOffset, u32 firstInstance) const
    {
        vkCmdDrawIndexed(m_commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }

    IFRIT_APIDECL void CommandBuffer::Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const
    {
        vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
    }

    IFRIT_APIDECL void CommandBuffer::DrawMeshTasks(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const
    {
        m_context->GetExtensionFunction().p_vkCmdDrawMeshTasksEXT(
            m_commandBuffer, groupCountX, groupCountY, groupCountZ);
    }

    IFRIT_APIDECL void CommandBuffer::DrawMeshTasksIndirect(
        const Rhi::RhiBuffer* buffer, u32 offset, u32 drawCount, u32 stride) const
    {
        auto buf = CheckedCast<SingleBuffer>(buffer)->GetBuffer();
        m_context->GetExtensionFunction().p_vkCmdDrawMeshTasksIndirectEXT(
            m_commandBuffer, buf, offset, drawCount, stride);
    }

    IFRIT_APIDECL void CommandBuffer::CopyBuffer(
        const Rhi::RhiBuffer* srcBuffer, const Rhi::RhiBuffer* dstBuffer, u32 size, u32 srcOffset, u32 dstOffset) const
    {
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = srcOffset;
        copyRegion.dstOffset = dstOffset;
        copyRegion.size      = size;
        auto src             = CheckedCast<SingleBuffer>(srcBuffer)->GetBuffer();
        auto dst             = CheckedCast<SingleBuffer>(dstBuffer)->GetBuffer();
        vkCmdCopyBuffer(m_commandBuffer, src, dst, 1, &copyRegion);
    }

    IFRIT_APIDECL void CommandBuffer::CopyBufferToImageAllInternal(const Rhi::RhiBuffer* srcBuffer, VkImage dstImage,
        VkImageLayout dstLayout, u32 width, u32 height, u32 depth) const
    {
        VkBufferImageCopy region{};
        region.bufferOffset                    = 0;
        region.bufferRowLength                 = 0;
        region.bufferImageHeight               = 0;
        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;
        region.imageOffset                     = { 0, 0, 0 };
        region.imageExtent                     = { width, height, depth };

        auto src = CheckedCast<SingleBuffer>(srcBuffer)->GetBuffer();
        vkCmdCopyBufferToImage(m_commandBuffer, src, dstImage, dstLayout, 1, &region);
    }

    IFRIT_APIDECL void CommandBuffer::CopyBufferToImage(
        const Rhi::RhiBuffer* src, const Rhi::RhiTexture* dst, Rhi::RhiImageSubResource dstSub) const
    {
        auto              image = CheckedCast<SingleDeviceImage>(dst);
        VkBufferImageCopy region{};
        region.bufferOffset                    = 0;
        region.bufferRowLength                 = 0;
        region.bufferImageHeight               = 0;
        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = dstSub.mipLevel;
        region.imageSubresource.baseArrayLayer = dstSub.arrayLayer;
        region.imageSubresource.layerCount     = dstSub.layerCount;
        region.imageOffset                     = { 0, 0, 0 };
        region.imageExtent                     = { image->GetWidth(), image->GetHeight(), image->GetDepth() };

        auto buffer = CheckedCast<SingleBuffer>(src)->GetBuffer();
        vkCmdCopyBufferToImage(
            m_commandBuffer, buffer, image->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    IFRIT_APIDECL void CommandBuffer::GlobalMemoryBarrier() const
    {
        VkMemoryBarrier barrier{};
        barrier.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            1, &barrier, 0, nullptr, 0, nullptr);
    }

    // Rhi compatible
    IFRIT_APIDECL void CommandBuffer::AddImageBarrier(Rhi::RhiTexture* texture, Rhi::RhiResourceState src,
        Rhi::RhiResourceState dst, Rhi::RhiImageSubResource subResource) const
    {
        auto                 image = CheckedCast<SingleDeviceImage>(texture);
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        // select old layout

        _setTextureState(image, dst);
        // if auto traced, we need to get the current state
        switch (src)
        {
            case Rhi::RhiResourceState::Undefined:
                barrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_HOST_READ_BIT
                    | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                    | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                    | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_MEMORY_READ_BIT
                    | VK_ACCESS_MEMORY_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::Common:
                barrier.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
                    | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                    | VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case Rhi::RhiResourceState::ShaderRead:
                barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
                    | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                    | VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case Rhi::RhiResourceState::ColorRT:
                barrier.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::Present:
                barrier.oldLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                break;
            case Rhi::RhiResourceState::UnorderedAccess:
                barrier.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                break;
            case Rhi::RhiResourceState::DepthStencilRT:
                barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                barrier.srcAccessMask =
                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                break;
            case Rhi::RhiResourceState::CopySrc:
                barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case Rhi::RhiResourceState::CopyDst:
                barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                break;

            default:
                vkrError("Invalid resource state");
        }
        // select new layout
        switch (dst)
        {
            case Rhi::RhiResourceState::Undefined:
                barrier.newLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.dstAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_HOST_READ_BIT
                    | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                    | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                    | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_MEMORY_READ_BIT
                    | VK_ACCESS_MEMORY_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::Common:
                barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
                    | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                    | VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case Rhi::RhiResourceState::ShaderRead:
                barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
                    | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                    | VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case Rhi::RhiResourceState::ColorRT:
                barrier.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::DepthStencilRT:
                barrier.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::Present:
                barrier.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                break;
            case Rhi::RhiResourceState::UnorderedAccess:
                barrier.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
                barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                break;

            case Rhi::RhiResourceState::CopySrc:
                barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;

            case Rhi::RhiResourceState::CopyDst:
                barrier.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;
            default:
                vkrError("Invalid resource state");
        }
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        auto p                                  = image->GetImage();
        barrier.image                           = p;
        barrier.subresourceRange.aspectMask     = image->GetAspect();
        barrier.subresourceRange.layerCount     = subResource.layerCount;
        barrier.subresourceRange.levelCount     = subResource.mipCount;
        barrier.subresourceRange.baseArrayLayer = subResource.arrayLayer;
        barrier.subresourceRange.baseMipLevel   = subResource.mipLevel;

        vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            0, nullptr, 0, nullptr, 1, &barrier);
    }

    IFRIT_APIDECL void CommandBuffer::AttachUniformRef(u32 setId, Rhi::RhiBindlessDescriptorRef* ref) const
    {
        if (m_CmdContext->m_BoundType == CommandListContextPrivate::BoundType::Graphics)
        {
            auto bindless     = CheckedCast<DescriptorBindlessIndices>(ref);
            auto graphicsPass = CheckedCast<GraphicsPass>(m_CmdContext->m_BoundPass.m_GraphicsPass);
            auto set          = bindless->GetActiveRangeSet();
            auto offset       = bindless->GetActiveRangeOffset();

            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPass->GetPipelineLayout(),
                setId, 1, &set, 1, &offset);
        }
        else if (m_CmdContext->m_BoundType == CommandListContextPrivate::BoundType::Compute)
        {
            auto bindless    = CheckedCast<DescriptorBindlessIndices>(ref);
            auto computePass = CheckedCast<ComputePass>(m_CmdContext->m_BoundPass.m_ComputePass);
            auto set         = bindless->GetActiveRangeSet();
            auto offset      = bindless->GetActiveRangeOffset();

            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePass->GetPipelineLayout(),
                setId, 1, &set, 1, &offset);
        }
        else
        {
            vkrError("Invalid command buffer bound type");
        }
    }

    IFRIT_APIDECL void CommandBuffer::AttachVertexBufferView(const Rhi::RhiVertexBufferView& view) const
    {
        auto vxDesc = CheckedCast<VertexBufferDescriptor>(&view);
        auto exfun  = m_context->GetExtensionFunction();
        exfun.p_vkCmdSetVertexInputEXT(m_commandBuffer, SizeCast<u32>(vxDesc->m_bindings.size()),
            vxDesc->m_bindings.data(), SizeCast<u32>(vxDesc->m_attributes.size()), vxDesc->m_attributes.data());
    }

    IFRIT_APIDECL void CommandBuffer::AttachVertexBuffers(u32 firstSlot, const Vec<Rhi::RhiBuffer*>& buffers) const
    {
        Vec<VkBuffer>     vxbuffers;
        Vec<VkDeviceSize> offsets;
        for (int i = 0; i < buffers.size(); i++)
        {
            auto buffer = CheckedCast<SingleBuffer>(buffers[i]);
            vxbuffers.push_back(buffer->GetBuffer());
            offsets.push_back(0);
        }
        vkCmdBindVertexBuffers(
            m_commandBuffer, firstSlot, SizeCast<u32>(buffers.size()), vxbuffers.data(), offsets.data());
    }

    IFRIT_APIDECL void CommandBuffer::AttachIndexBuffer(const Rhi::RhiBuffer* buffer) const
    {
        auto buf = CheckedCast<SingleBuffer>(buffer)->GetBuffer();
        vkCmdBindIndexBuffer(m_commandBuffer, buf, 0, VK_INDEX_TYPE_UINT32);
    }

    IFRIT_APIDECL void CommandBuffer::DrawInstanced(
        u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const
    {
        vkCmdDraw(m_commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance);
    }

    IFRIT_APIDECL void CommandBuffer::BufferClear(const Rhi::RhiBuffer* buffer, u32 val) const
    {
        auto buf = CheckedCast<SingleBuffer>(buffer)->GetBuffer();
        vkCmdFillBuffer(m_commandBuffer, buf, 0, VK_WHOLE_SIZE, val);
    }

    IFRIT_APIDECL void CommandBuffer::DispatchIndirect(const Rhi::RhiBuffer* buffer, u32 offset) const
    {
        auto buf = CheckedCast<SingleBuffer>(buffer)->GetBuffer();
        vkCmdDispatchIndirect(m_commandBuffer, buf, offset);
    }

    IFRIT_APIDECL void CommandBuffer::SetPushConst(const void* data, u32 offset, u32 size) const
    {
        if (m_CmdContext->m_BoundType == CommandListContextPrivate::BoundType::Graphics)
        {
            auto graphicsPass = CheckedCast<GraphicsPass>(m_CmdContext->m_BoundPass.m_GraphicsPass);
            vkCmdPushConstants(
                m_commandBuffer, graphicsPass->GetPipelineLayout(), VK_SHADER_STAGE_ALL, offset, size, data);
            return;
        }
        else if (m_CmdContext->m_BoundType == CommandListContextPrivate::BoundType::Compute)
        {
            auto computePass = CheckedCast<ComputePass>(m_CmdContext->m_BoundPass.m_ComputePass);
            vkCmdPushConstants(
                m_commandBuffer, computePass->GetPipelineLayout(), VK_SHADER_STAGE_ALL, offset, size, data);
            return;
        }
        else
        {
            vkrError("Invalid command buffer bound type");
        }
    };

    IFRIT_APIDECL void CommandBuffer::ClearUAVTexFloat(
        const Rhi::RhiTexture* texture, Rhi::RhiImageSubResource subResource, const std::array<float, 4>& val) const
    {
        auto              image = CheckedCast<SingleDeviceImage>(texture);
        VkClearColorValue clearColor;
        clearColor.float32[0] = val[0];
        clearColor.float32[1] = val[1];
        clearColor.float32[2] = val[2];
        clearColor.float32[3] = val[3];
        VkImageSubresourceRange range{};
        range.aspectMask     = image->GetAspect();
        range.baseMipLevel   = subResource.mipLevel;
        range.levelCount     = subResource.mipCount;
        range.baseArrayLayer = subResource.arrayLayer;
        range.layerCount     = subResource.layerCount;
        vkCmdClearColorImage(m_commandBuffer, image->GetImage(), VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
    }

    IFRIT_APIDECL void CommandBuffer::CopyImage(const Rhi::RhiTexture* src, Rhi::RhiImageSubResource srcSub,
        const Rhi::RhiTexture* dst, Rhi::RhiImageSubResource dstSub) const
    {
        auto        srcImage = CheckedCast<SingleDeviceImage>(src);
        auto        dstImage = CheckedCast<SingleDeviceImage>(dst);
        VkImageCopy region{};
        region.srcSubresource.aspectMask     = srcImage->GetAspect();
        region.srcSubresource.mipLevel       = srcSub.mipLevel;
        region.srcSubresource.baseArrayLayer = srcSub.arrayLayer;
        region.srcSubresource.layerCount     = srcSub.layerCount;
        region.srcOffset                     = { 0, 0, 0 };
        region.dstSubresource.aspectMask     = dstImage->GetAspect();
        region.dstSubresource.mipLevel       = dstSub.mipLevel;
        region.dstSubresource.baseArrayLayer = dstSub.arrayLayer;
        region.dstSubresource.layerCount     = dstSub.layerCount;
        region.dstOffset                     = { 0, 0, 0 };
        region.extent.width                  = srcImage->GetWidth();
        region.extent.height                 = srcImage->GetHeight();
        region.extent.depth                  = srcImage->GetDepth();
        vkCmdCopyImage(m_commandBuffer, srcImage->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dstImage->GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    };

    void _resourceStateToAccessMask(Rhi::RhiResourceState state, VkAccessFlags& dstAccessMask)
    {
        switch (state)
        {
            case Rhi::RhiResourceState::Undefined:
                dstAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
                    | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT
                    | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                    | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                    | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::Common:
                dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT
                    | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case Rhi::RhiResourceState::ColorRT:
                dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::Present:
                dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
                break;
            case Rhi::RhiResourceState::DepthStencilRT:
                dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::UnorderedAccess:
                dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                break;
            case Rhi::RhiResourceState::CopySrc:
                dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case Rhi::RhiResourceState::CopyDst:
                dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;
            case Rhi::RhiResourceState::ShaderRead:
                dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                break;
            default:
                vkrError("Invalid resource state");
        }
    }

    void _resourceStateToImageLayout(Rhi::RhiResourceState state, VkImageLayout& dstLayout)
    {
        switch (state)
        {
            case Rhi::RhiResourceState::Undefined:
                dstLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                break;
            case Rhi::RhiResourceState::Common:
                dstLayout = VK_IMAGE_LAYOUT_GENERAL;
                break;
            case Rhi::RhiResourceState::ColorRT:
                dstLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                break;
            case Rhi::RhiResourceState::Present:
                dstLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
                break;
            case Rhi::RhiResourceState::DepthStencilRT:
                dstLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                break;
            case Rhi::RhiResourceState::UnorderedAccess:
                dstLayout = VK_IMAGE_LAYOUT_GENERAL;
                break;
            case Rhi::RhiResourceState::CopySrc:
                dstLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                break;
            case Rhi::RhiResourceState::CopyDst:
                dstLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                break;
            case Rhi::RhiResourceState::ShaderRead:
                dstLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                break;
            default:
                vkrError("Invalid resource state");
        }
    }

    IFRIT_APIDECL
    void CommandBuffer::AddResourceBarrier(const Vec<Rhi::RhiResourceBarrier>& barriers) const
    {
        Vec<VkImageMemoryBarrier>  imageBarriers;
        Vec<VkBufferMemoryBarrier> bufferBarriers;
        for (int i = 0; i < barriers.size(); i++)
        {
            auto barrier = barriers[i];
            if (barrier.m_type == Rhi::RhiBarrierType::Transition)
            {
                auto resourceType = barrier.m_transition.m_type;
                auto srcState     = barrier.m_transition.m_srcState;
                if (srcState == Rhi::RhiResourceState::AutoTraced)
                {
                    if (barrier.m_transition.m_type == Rhi::RhiResourceType::Texture)
                    {
                        srcState = barrier.m_transition.m_texture->GetState();
                    }
                    else
                    {
                        srcState = barrier.m_transition.m_buffer->GetState();
                    }
                }
                else
                {
                    iError("Source state is not auto traced");
                    std::abort();
                }
                auto          dstState = barrier.m_transition.m_dstState;
                VkAccessFlags srcAccessMask, dstAccessMask;
                _resourceStateToAccessMask(srcState, srcAccessMask);
                _resourceStateToAccessMask(dstState, dstAccessMask);

                if (resourceType == Rhi::RhiResourceType::Buffer)
                {
                    VkBufferMemoryBarrier bufferBarrier{};
                    bufferBarrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    bufferBarrier.buffer        = CheckedCast<SingleBuffer>(barrier.m_transition.m_buffer)->GetBuffer();
                    bufferBarrier.offset        = 0;
                    bufferBarrier.size          = VK_WHOLE_SIZE;
                    bufferBarrier.srcAccessMask = srcAccessMask;
                    bufferBarrier.dstAccessMask = dstAccessMask;
                    bufferBarriers.push_back(bufferBarrier);
                }
                else if (resourceType == Rhi::RhiResourceType::Texture)
                {
                    // WARNING: subresource unspecified
                    VkImageLayout srcLayout, dstLayout;
                    auto          subResource = barrier.m_transition.m_subResource;
                    _resourceStateToImageLayout(srcState, srcLayout);
                    _resourceStateToImageLayout(dstState, dstLayout);

                    if (srcState != barrier.m_transition.m_texture->GetState()
                        && srcState != Rhi::RhiResourceState::Undefined)
                    {
                        iError("Texture state mismatch, expected:{} actual:{}", i32(srcState),
                            i32(barrier.m_transition.m_texture->GetState()));

                        std::abort();
                    }
                    _setTextureState(barrier.m_transition.m_texture, dstState);
                    auto                 image  = CheckedCast<SingleDeviceImage>(barrier.m_transition.m_texture);
                    auto                 aspect = image->GetAspect();
                    VkImageMemoryBarrier imageBarrier{};
                    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imageBarrier.image = CheckedCast<SingleDeviceImage>(barrier.m_transition.m_texture)->GetImage();
                    imageBarrier.subresourceRange.aspectMask     = aspect;
                    imageBarrier.subresourceRange.baseMipLevel   = subResource.mipLevel;
                    imageBarrier.subresourceRange.levelCount     = subResource.mipCount;
                    imageBarrier.subresourceRange.baseArrayLayer = subResource.arrayLayer;
                    imageBarrier.subresourceRange.layerCount     = subResource.layerCount;
                    imageBarrier.srcAccessMask                   = srcAccessMask;
                    imageBarrier.dstAccessMask                   = dstAccessMask;
                    imageBarrier.oldLayout                       = srcLayout;
                    imageBarrier.newLayout                       = dstLayout;
                    imageBarriers.push_back(imageBarrier);
                }
            }
            else if (barrier.m_type == Rhi::RhiBarrierType::UAVAccess)
            {
                auto resourceType = barrier.m_uav.m_type;
                if (resourceType == Rhi::RhiResourceType::Buffer)
                {
                    VkBufferMemoryBarrier bufferBarrier{};
                    bufferBarrier.sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    bufferBarrier.buffer        = CheckedCast<SingleBuffer>(barrier.m_uav.m_buffer)->GetBuffer();
                    bufferBarrier.offset        = 0;
                    bufferBarrier.size          = VK_WHOLE_SIZE;
                    bufferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    bufferBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    bufferBarriers.push_back(bufferBarrier);
                }
                else if (resourceType == Rhi::RhiResourceType::Texture)
                {

                    if (barrier.m_uav.m_texture->GetState() != Rhi::RhiResourceState::Common)
                    {
                        iError("Texture state mismatch, expected:{} actual:{}", i32(Rhi::RhiResourceState::Common),
                            i32(barrier.m_uav.m_texture->GetState()));
                        std::abort();
                    }
                    _setTextureState(barrier.m_uav.m_texture, Rhi::RhiResourceState::Common);

                    // WARNING: subresource unspecified
                    VkImageMemoryBarrier imageBarrier{};
                    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imageBarrier.image = CheckedCast<SingleDeviceImage>(barrier.m_uav.m_texture)->GetImage();
                    imageBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
                    imageBarrier.subresourceRange.baseMipLevel   = 0;
                    imageBarrier.subresourceRange.levelCount     = 1;
                    imageBarrier.subresourceRange.baseArrayLayer = 0;
                    imageBarrier.subresourceRange.layerCount     = 1;
                    imageBarrier.srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
                    imageBarrier.dstAccessMask                   = VK_ACCESS_SHADER_READ_BIT;
                    imageBarrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
                    imageBarrier.newLayout                       = VK_IMAGE_LAYOUT_GENERAL;
                    imageBarriers.push_back(imageBarrier);
                }
            }
        }
        vkCmdPipelineBarrier(m_commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            0, nullptr, SizeCast<int>(bufferBarriers.size()), bufferBarriers.data(),
            SizeCast<int>(imageBarriers.size()), imageBarriers.data());
    }

    IFRIT_APIDECL void CommandBuffer::BeginScope(const std::string& name) const
    {
        VkDebugUtilsLabelEXT label{};
        label.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        label.pLabelName = name.c_str();
        label.color[0]   = 1.0f;
        label.color[1]   = 1.0f;
        label.color[2]   = 1.0f;
        label.color[3]   = 1.0f;
        auto exfun       = m_context->GetExtensionFunction();
        if (m_context->IsDebugMode())
            exfun.p_vkCmdBeginDebugUtilsLabelEXT(m_commandBuffer, &label);
    }
    IFRIT_APIDECL void CommandBuffer::EndScope() const
    {
        auto exfun = m_context->GetExtensionFunction();
        if (m_context->IsDebugMode())
            exfun.p_vkCmdEndDebugUtilsLabelEXT(m_commandBuffer);
    }

    IFRIT_APIDECL void CommandBuffer::SetCullMode(Rhi::RhiCullMode mode) const
    {
        VkCullModeFlags cullMode;
        switch (mode)
        {
            case Rhi::RhiCullMode::None:
                cullMode = VK_CULL_MODE_NONE;
                break;
            case Rhi::RhiCullMode::Front:
                cullMode = VK_CULL_MODE_FRONT_BIT;
                break;
            case Rhi::RhiCullMode::Back:
                cullMode = VK_CULL_MODE_BACK_BIT;
                break;
            default:
                vkrError("Invalid cull mode");
        }
        auto exfun = m_context->GetExtensionFunction();
        vkCmdSetCullMode(m_commandBuffer, cullMode);
    }
    // Class: Queue
    IFRIT_APIDECL Queue::Queue(
        EngineContext* ctx, VkQueue queue, u32 family, VkQueueFlags capability, u32 m_InFlightFrames)
        : m_context(ctx)
        , m_queue(queue)
        , m_queueFamily(family)
        , m_capability(capability)
        , m_InFlightFrames(m_InFlightFrames)
    {
        // m_commandPool       = std::make_unique<CommandPool>(ctx, family);
        for (u32 i = 0; i < m_InFlightFrames; i++)
        {
            m_commandPools.push_back(std::make_unique<CommandPool>(ctx, family));
        }
        m_timelineSemaphore = std::make_unique<TimelineSemaphore>(ctx);
    }

    IFRIT_APIDECL void Queue::FrameAdvance()
    {
        m_ActiveFrame    = (m_ActiveFrame + 1) % m_InFlightFrames;
        auto currentPool = m_commandPools[m_ActiveFrame].get();
        currentPool->ResetCommandPool();
    }

    IFRIT_APIDECL CommandBuffer* Queue::BeginRecording()
    {
        if (m_cmdBufInUse.size() != 0)
        {
            vkrError("Command buffer still in use");
        }
        Uref<CommandBuffer> buffer;
        buffer = m_commandPools[m_ActiveFrame]->AllocateCommandBufferUnique();

        if (buffer == nullptr)
        {
            vkrError("Nullptr");
        }
        auto p                 = buffer.get();
        m_currentCommandBuffer = p;
        m_cmdBufInUse.push(std::move(buffer));

        p->BeginRecord();
        return p;
    }

    IFRIT_APIDECL TimelineSemaphoreWait Queue::SubmitCommand(
        const Vec<TimelineSemaphoreWait>& waitSemaphores, VkFence fence, VkSemaphore swapchainSemaphore)
    {
        m_recordedCounter++;
        Vec<VkSemaphore>          waitSemaphoreHandles;
        Vec<uint64_t>             waitValues;
        Vec<VkPipelineStageFlags> waitStages;

        for (int i = 0; i < waitSemaphores.size(); i++)
        {
            waitSemaphoreHandles.push_back(waitSemaphores[i].m_semaphore);
            waitValues.push_back(waitSemaphores[i].m_value);
            waitStages.push_back(waitSemaphores[i].m_waitStage);
        }

        Vec<uint64_t>    signalValues;
        Vec<VkSemaphore> signalSemaphores;
        signalValues.push_back(m_recordedCounter);
        signalSemaphores.push_back(m_timelineSemaphore->getSemaphore());

        if (swapchainSemaphore)
        {
            signalValues.push_back(0);
            signalSemaphores.push_back(swapchainSemaphore);
        }

        VkCommandBuffer commandBuffer = m_currentCommandBuffer->GetCommandBuffer();
        m_currentCommandBuffer->EndRecord();

        VkSubmitInfo submitInfo{};
        submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount   = SizeCast<int>(waitSemaphoreHandles.size());
        submitInfo.pWaitSemaphores      = waitSemaphoreHandles.data();
        submitInfo.pWaitDstStageMask    = waitStages.data();
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &commandBuffer;
        submitInfo.signalSemaphoreCount = SizeCast<int>(signalSemaphores.size());
        submitInfo.pSignalSemaphores    = signalSemaphores.data();

        VkTimelineSemaphoreSubmitInfo timelineInfo{};
        timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineInfo.waitSemaphoreValueCount   = SizeCast<int>(waitValues.size());
        timelineInfo.pWaitSemaphoreValues      = waitValues.data();
        timelineInfo.signalSemaphoreValueCount = SizeCast<int>(signalValues.size());
        timelineInfo.pSignalSemaphoreValues    = signalValues.data();

        submitInfo.pNext = &timelineInfo;
        VkFence vfence   = VK_NULL_HANDLE;
        if (fence)
        {
            vfence = fence;
        }
        vkrVulkanAssert(vkQueueSubmit(m_queue, 1, &submitInfo, vfence), "Failed to submit command buffer");
        // move the command buffer to the free list
        Uref<CommandBuffer> cmdBuf = std::move(m_cmdBufInUse.top());
        m_commandPools[m_ActiveFrame]->EnqueueInFlightCommandBuffer(std::move(cmdBuf));
        m_cmdBufInUse.pop();

        TimelineSemaphoreWait ret;
        ret.m_semaphore = m_timelineSemaphore.get()->getSemaphore();
        ret.m_value     = m_recordedCounter;
        ret.m_waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        return ret;
    }

    IFRIT_APIDECL void Queue::WaitIdle() { vkQueueWaitIdle(m_queue); }

    IFRIT_APIDECL void Queue::CounterReset() { m_recordedCounter = 0; }

    // Class: CommandSubmissionList
    IFRIT_APIDECL      CommandSubmissionList::CommandSubmissionList(EngineContext* ctx) : m_context(ctx)
    {
        m_hostSyncSemaphore = std::make_unique<TimelineSemaphore>(ctx);
    }

    IFRIT_APIDECL void CommandSubmissionList::AddSubmission(const CommandSubmissionInfo& info)
    {
        m_submissions.push_back(info);
    }

    IFRIT_APIDECL void CommandSubmissionList::Submit(bool hostSync)
    {
        for (int i = 0; i < m_submissions.size(); i++)
        {
            auto&                     submission = m_submissions[i];
            Vec<VkSemaphore>          waitSemaphores;
            Vec<VkPipelineStageFlags> waitStages;
            Vec<VkSemaphore>          signalSemaphores;
            Vec<uint64_t>             signalValues;
            Vec<uint64_t>             waitValues;

            for (int j = 0; j < submission.m_waitSemaphore.size(); j++)
            {
                waitSemaphores.push_back(submission.m_waitSemaphore[j]->getSemaphore());
                waitStages.push_back(submission.m_waitStages[j]);
                waitValues.push_back(submission.m_waitValues[j]);
            }
            for (int j = 0; j < submission.m_signalSemaphore.size(); j++)
            {
                signalSemaphores.push_back(submission.m_signalSemaphore[j]->getSemaphore());
                signalValues.push_back(submission.m_signalValues[j]);
            }

            if (hostSync)
            {
                waitSemaphores.push_back(m_hostSyncSemaphore->getSemaphore());
                waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                waitValues.push_back(1);
            }

            VkSubmitInfo          submitInfo{};
            const VkCommandBuffer commandBuffer = submission.m_commandBuffer->GetCommandBuffer();
            submitInfo.sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.waitSemaphoreCount       = SizeCast<int>(waitSemaphores.size());
            submitInfo.pWaitSemaphores          = waitSemaphores.data();
            submitInfo.pWaitDstStageMask        = waitStages.data();
            submitInfo.commandBufferCount       = 1;
            submitInfo.pCommandBuffers          = &commandBuffer;
            submitInfo.signalSemaphoreCount     = SizeCast<int>(signalSemaphores.size());
            submitInfo.pSignalSemaphores        = signalSemaphores.data();

            VkTimelineSemaphoreSubmitInfo timelineInfo{};
            timelineInfo.sType                     = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timelineInfo.waitSemaphoreValueCount   = SizeCast<int>(waitValues.size());
            timelineInfo.pWaitSemaphoreValues      = waitValues.data();
            timelineInfo.signalSemaphoreValueCount = SizeCast<int>(signalValues.size());
            timelineInfo.pSignalSemaphoreValues    = signalValues.data();

            submitInfo.pNext = &timelineInfo;

            vkrVulkanAssert(vkQueueSubmit(submission.m_queue->GetQueue(), 1, &submitInfo, VK_NULL_HANDLE),
                "Failed to submit command buffer");
        }
    }

    void Queue::RunSyncCommand(std::function<void(const Rhi::RhiCommandList*)> func)
    {
        auto cmd = BeginRecording();
        func(cmd);
        SubmitCommand({}, VK_NULL_HANDLE);
        WaitIdle();
    }

    std::unique_ptr<Rhi::RhiTaskSubmission> Queue::RunAsyncCommand(std::function<void(const Rhi::RhiCommandList*)> func,
        const Vec<Rhi::RhiTaskSubmission*>& waitOn, const Vec<Rhi::RhiTaskSubmission*>& toIssue)
    {
        auto cmd = BeginRecording();
        func(cmd);
        Vec<TimelineSemaphoreWait> waitSemaphores;

        VkSemaphore                swapchainSemaphore = VK_NULL_HANDLE;
        VkFence                    fence              = VK_NULL_HANDLE;
        if (toIssue.size() != 0)
        {
            vkrAssert(toIssue.size() == 1, "Only one semaphore can be issued");
            auto semaphore = CheckedCast<TimelineSemaphoreWait>(toIssue[0]);
            vkrAssert(semaphore->m_isSwapchainSemaphore, "Only swapchain semaphore can be issued");
            swapchainSemaphore = semaphore->m_semaphore;
            fence              = semaphore->m_fence;
        }
        for (int i = 0; i < waitOn.size(); i++)
        {
            auto semaphore = CheckedCast<TimelineSemaphoreWait>(waitOn[i]);
            waitSemaphores.push_back(*semaphore);
        }
        return std::make_unique<TimelineSemaphoreWait>(SubmitCommand(waitSemaphores, fence, swapchainSemaphore));
    }

    void Queue::HostWaitEvent(Rhi::RhiTaskSubmission* event)
    {
        VkSemaphoreWaitInfo waitInfo{};
        auto                sev = CheckedCast<TimelineSemaphoreWait>(event);
        waitInfo.sType          = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores    = &sev->m_semaphore;
        waitInfo.pValues        = &sev->m_value;
        vkrVulkanAssert(vkWaitSemaphores(m_context->GetDevice(), &waitInfo, std::numeric_limits<uint64_t>::max()),
            "Failed to wait for semaphore");
    }

    // Queue Collections
    IFRIT_APIDECL void QueueCollections::LoadQueues(u32 numFramesInFlight)
    {
        auto& queueData = m_context->GetQueueInfo();
        for (int i = 0; i < queueData.m_allQueues.size(); i++)
        {
            auto queue           = queueData.m_allQueues[i];
            auto queueCapability = queueData.m_queueFamilies[queue.m_familyIndex].m_capability;
            m_queues.push_back(std::make_unique<Queue>(
                m_context, queue.m_queue, queue.m_familyIndex, queueCapability, numFramesInFlight));
        }
    }

    IFRIT_APIDECL void QueueCollections::FrameAdvance()
    {
        for (int i = 0; i < m_queues.size(); i++)
        {
            m_queues[i]->FrameAdvance();
        }
    }

    IFRIT_APIDECL Vec<Queue*> QueueCollections::GetGraphicsQueues()
    {
        Vec<Queue*> graphicsQueues;
        for (int i = 0; i < m_queues.size(); i++)
        {
            if (m_queues[i]->GetCapability() & VK_QUEUE_GRAPHICS_BIT)
            {
                graphicsQueues.push_back(m_queues[i].get());
            }
        }
        return graphicsQueues;
    }

    IFRIT_APIDECL Vec<Queue*> QueueCollections::GetComputeQueues()
    {
        Vec<Queue*> computeQueues;
        for (int i = 0; i < m_queues.size(); i++)
        {
            if (m_queues[i]->GetCapability() & VK_QUEUE_COMPUTE_BIT)
            {
                computeQueues.push_back(m_queues[i].get());
            }
        }
        return computeQueues;
    }

    IFRIT_APIDECL Vec<Queue*> QueueCollections::GetTransferQueues()
    {
        Vec<Queue*> transferQueues;
        for (int i = 0; i < m_queues.size(); i++)
        {
            if (m_queues[i]->GetCapability() & VK_QUEUE_COMPUTE_BIT)
            {
                continue;
            }
            if (m_queues[i]->GetCapability() & VK_QUEUE_TRANSFER_BIT)
            {
                transferQueues.push_back(m_queues[i].get());
            }
        }
        return transferQueues;
    }

} // namespace Ifrit::Graphics::VulkanGraphics