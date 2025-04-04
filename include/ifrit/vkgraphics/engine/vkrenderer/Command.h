
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

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <memory>
#include <vector>
#include <stack>

namespace Ifrit::Graphics::VulkanGraphics
{

    class CommandBuffer;

    class IFRIT_APIDECL VertexBufferDescriptor : public Rhi::RhiVertexBufferView
    {
    public:
        Vec<VkVertexInputAttributeDescription2EXT> m_attributes;
        Vec<VkVertexInputBindingDescription2EXT>   m_bindings;
        inline void AddBinding(Vec<u32> location, Vec<Rhi::RhiImageFormat> format, Vec<u32> offset, u32 stride,
            Rhi::RhiVertexInputRate inputRate = Rhi::RhiVertexInputRate::Vertex) override
        {
            VkVertexInputBindingDescription2EXT binding{};
            binding.binding = Ifrit::SizeCast<u32>(m_bindings.size());
            binding.stride  = stride;
            binding.divisor = 1;
            binding.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT;

            if (inputRate == Rhi::RhiVertexInputRate::Vertex)
            {
                binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
            }
            else
            {
                binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
            }
            m_bindings.push_back(binding);

            for (i32 i = 0; i < location.size(); i++)
            {
                VkVertexInputAttributeDescription2EXT attribute{};
                attribute.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT;
                attribute.binding  = binding.binding;
                attribute.format   = static_cast<VkFormat>(format[i]);
                attribute.location = location[i];
                attribute.offset   = offset[i];
                m_attributes.push_back(attribute);
            }
        }
    };

    class IFRIT_APIDECL TimelineSemaphore
    {
    private:
        EngineContext* m_context;
        VkSemaphore    m_semaphore;
        u64            m_recordedCounter = 0;

    public:
        TimelineSemaphore(EngineContext* ctx);
        ~TimelineSemaphore();
        inline VkSemaphore getSemaphore() const { return m_semaphore; }
    };

    class TimelineSemaphoreWait : public Rhi::RhiTaskSubmission
    {
    public:
        VkSemaphore            m_semaphore;
        VkFence                m_fence;
        u64                    m_value;
        VkFlags                m_waitStage            = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        bool                   m_isSwapchainSemaphore = false;
        TimelineSemaphoreWait& operator=(const TimelineSemaphoreWait& other)
        {
            m_semaphore            = other.m_semaphore;
            m_value                = other.m_value;
            m_waitStage            = other.m_waitStage;
            m_isSwapchainSemaphore = other.m_isSwapchainSemaphore;
            return *this;
        }
    };

    class IFRIT_APIDECL PipelineBarrier
    {
    private:
        EngineContext*             m_context;
        VkPipelineStageFlags       m_srcStage;
        VkPipelineStageFlags       m_dstStage;
        VkDependencyFlags          m_dependencyFlags;
        Vec<VkMemoryBarrier>       m_memoryBarriers;
        Vec<VkBufferMemoryBarrier> m_bufferMemoryBarriers;
        Vec<VkImageMemoryBarrier>  m_imageMemoryBarriers;

    public:
        PipelineBarrier(EngineContext* ctx, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
            VkDependencyFlags dependencyFlags)
            : m_context(ctx), m_srcStage(srcStage), m_dstStage(dstStage), m_dependencyFlags(dependencyFlags)
        {
        }

        void addMemoryBarrier(VkMemoryBarrier barrier);
        void addBufferMemoryBarrier(VkBufferMemoryBarrier barrier);
        void addImageMemoryBarrier(VkImageMemoryBarrier barrier);

        friend class CommandBuffer;
    };

    class IFRIT_APIDECL CommandBuffer : public Rhi::RhiCommandList
    {
    private:
        EngineContext*  m_context;
        VkCommandBuffer m_commandBuffer;
        u32             m_queueFamily;

    public:
        CommandBuffer(EngineContext* ctx, VkCommandBuffer buffer, u32 queueFamily)
            : m_context(ctx), m_commandBuffer(buffer), m_queueFamily(queueFamily)
        {
        }
        virtual ~CommandBuffer() {}

        inline u32             GetQueueFamily() const { return m_queueFamily; }
        void                   BeginRecord();
        void                   EndRecord();
        void                   Reset();
        inline VkCommandBuffer GetCommandBuffer() const { return m_commandBuffer; }

        // Functionality
        void                   AddPipelineBarrier(const PipelineBarrier& barrier) const;

        void                   Draw(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const;
        void                   DrawMeshTasks(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const override;
        void                   DrawIndexed(
                              u32 indexCount, u32 instanceCount, u32 firstIndex, int32_t vertexOffset, u32 firstInstance) const override;

        void CopyBuffer(const Rhi::RhiBuffer* srcBuffer, const Rhi::RhiBuffer* dstBuffer, u32 size, u32 srcOffset = 0,
            u32 dstOffset = 0) const;
        void CopyBufferToImageAllInternal(const Rhi::RhiBuffer* srcBuffer, VkImage dstImage, VkImageLayout dstLayout,
            u32 width, u32 height, u32 depth) const;

        // Rhi compatible
        void SetViewports(const Vec<Rhi::RhiViewport>& viewport) const override;
        void SetScissors(const Vec<Rhi::RhiScissor>& scissor) const override;
        void Dispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) const override;
        void DrawMeshTasksIndirect(const Rhi::RhiBuffer* buffer, u32 offset, u32 drawCount, u32 stride) const override;

        void AddImageBarrier(Rhi::RhiTexture* texture, Rhi::RhiResourceState src, Rhi::RhiResourceState dst,
            Rhi::RhiImageSubResource subResource) const; // DEPRECATED

        void AttachBindlessRefGraphics(
            Rhi::RhiGraphicsPass* pass, u32 setId, Rhi::RhiBindlessDescriptorRef* ref) const override;
        void AttachBindlessRefCompute(
            Rhi::RhiComputePass* pass, u32 setId, Rhi::RhiBindlessDescriptorRef* ref) const override;
        void AttachVertexBufferView(const Rhi::RhiVertexBufferView& view) const override;
        void AttachVertexBuffers(u32 firstSlot, const Vec<Rhi::RhiBuffer*>& buffers) const override;
        void AttachIndexBuffer(const Rhi::RhiBuffer* buffer) const override;
        void DrawInstanced(u32 vertexCount, u32 instanceCount, u32 firstVertex, u32 firstInstance) const override;
        void BufferClear(const Rhi::RhiBuffer* buffer, u32 val) const override;
        void DispatchIndirect(const Rhi::RhiBuffer* buffer, u32 offset) const override;
        void SetPushConst(Rhi::RhiComputePass* pass, u32 offset, u32 size, const void* data) const override;
        void SetPushConst(Rhi::RhiGraphicsPass* pass, u32 offset, u32 size, const void* data) const override;
        void ClearUAVTexFloat(const Rhi::RhiTexture* texture, Rhi::RhiImageSubResource subResource,
            const Array<float, 4>& val) const override;
        void AddResourceBarrier(const Vec<Rhi::RhiResourceBarrier>& barriers) const override;

        void GlobalMemoryBarrier() const override;
        void BeginScope(const std::string& name) const override;
        void EndScope() const override;

        void CopyImage(const Rhi::RhiTexture* src, Rhi::RhiImageSubResource srcSub, const Rhi::RhiTexture* dst,
            Rhi::RhiImageSubResource dstSub) const override;
        void CopyBufferToImage(
            const Rhi::RhiBuffer* src, const Rhi::RhiTexture* dst, Rhi::RhiImageSubResource dstSub) const override;
        void SetCullMode(Rhi::RhiCullMode mode) const override;
    };

    class IFRIT_APIDECL CommandPool : NonCopyable
    {
    private:
        EngineContext*           m_context;
        u32                      m_queueFamily;
        VkCommandPool            m_commandPool;

        Vec<Uref<CommandBuffer>> m_AvailableCommandBuffers;
        Vec<Uref<CommandBuffer>> m_InFlightCommandBuffers;

    protected:
        void Init();

    public:
        CommandPool(EngineContext* ctx, u32 chosenQueueFamily) : m_context(ctx), m_queueFamily(chosenQueueFamily)
        {
            Init();
        }
        ~CommandPool();
        Ref<CommandBuffer>  AllocateCommandBuffer();
        Uref<CommandBuffer> AllocateCommandBufferUnique();
        void                ResetCommandPool();

        void                EnqueueInFlightCommandBuffer(Uref<CommandBuffer>&& cmdBuf);
    };

    // Note that command buffers should be recycled in order to avoid memory leaks.
    // https://developer.download.nvidia.com/gameworks/events/GDC2016/Vulkan_Essentials_GDC16_tlorach.pdf#page=15.00
    class IFRIT_APIDECL Queue : public Rhi::RhiQueue, NonCopyable
    {
    private:
        EngineContext*                  m_context;
        VkQueue                         m_queue;
        u32                             m_queueFamily;
        u32                             m_capability;
        Vec<Uref<CommandPool>>          m_commandPools;
        Uref<TimelineSemaphore>         m_timelineSemaphore;
        std::stack<Uref<CommandBuffer>> m_cmdBufInUse;
        u64                             m_recordedCounter      = 0;
        CommandBuffer*                  m_currentCommandBuffer = nullptr;

        u32                             m_InFlightFrames = 0;
        u32                             m_ActiveFrame    = 0; // The current frame that is being processed by the GPU.

    public:
        Queue() { printf("Runtime Error:queue\n"); }
        Queue(EngineContext* ctx, VkQueue queue, u32 queueFamily, u32 capability, u32 inFlightFrames);

        virtual ~Queue() {}
        inline VkQueue        GetQueue() const { return m_queue; }
        inline u32            GetQueueFamily() const { return m_queueFamily; }
        inline u32            GetCapability() const { return m_capability; }

        CommandBuffer*        BeginRecording();
        TimelineSemaphoreWait SubmitCommand(
            const Vec<TimelineSemaphoreWait>& waitSemaphores, VkFence fence, VkSemaphore swapchainSemaphore = nullptr);
        void                         WaitIdle();
        void                         CounterReset();
        void                         FrameAdvance();

        // for rhi layers override
        void                         RunSyncCommand(std::function<void(const Rhi::RhiCommandList*)> func) override;

        Uref<Rhi::RhiTaskSubmission> RunAsyncCommand(std::function<void(const Rhi::RhiCommandList*)> func,
            const Vec<Rhi::RhiTaskSubmission*>& waitOn, const Vec<Rhi::RhiTaskSubmission*>& toIssue) override;

        void                         HostWaitEvent(Rhi::RhiTaskSubmission* event) override;
    };

    class IFRIT_APIDECL QueueCollections
    {
    private:
        EngineContext*   m_context;
        Vec<Uref<Queue>> m_queues;

    public:
        QueueCollections(EngineContext* ctx) : m_context(ctx) {}
        QueueCollections(const QueueCollections& p)            = delete; // copy constructor
        QueueCollections& operator=(const QueueCollections& p) = delete;

        void              LoadQueues(u32 numFramesInFlight);
        void              FrameAdvance();
        Vec<Queue*>       GetGraphicsQueues();
        Vec<Queue*>       GetComputeQueues();
        Vec<Queue*>       GetTransferQueues();
    };

    struct CommandSubmissionInfo
    {
        CommandBuffer*          m_commandBuffer;
        Queue*                  m_queue;
        Vec<TimelineSemaphore*> m_waitSemaphore;
        Vec<u64>                m_waitValues;
        Vec<TimelineSemaphore*> m_signalSemaphore;
        Vec<u64>                m_signalValues;
        Vec<VkFlags>            m_waitStages;
    };

    class IFRIT_APIDECL CommandSubmissionList
    {
    private:
        EngineContext*             m_context;
        Vec<CommandSubmissionInfo> m_submissions;
        Uref<TimelineSemaphore>    m_hostSyncSemaphore = nullptr;

    public:
        CommandSubmissionList(EngineContext* ctx);
        void AddSubmission(const CommandSubmissionInfo& info);
        void Submit(bool hostSync = false);
    };
} // namespace Ifrit::Graphics::VulkanGraphics