#pragma once
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/adapter/Device.h"

namespace Ifrit::RHI::VulkanRHI2
{
    // VulkanRHI2 Goal:
    // 1. Multi-thread recording support
    // 2. Stateful -> Stateless (flexiblity for render graph)
    // 3. Hide queue exposure

    // VA_CommandListContext:
    // - Owned by each thread. When RhiCommandList activates, it acquires the available context
    // with respect to the pipeline type (graphics/compute). This is managed by device.
    // - abstraction of command list context, which handles child command buffers
    // submission, with orders (upload->execution->upload->execution...), w/ref to unreal
    // - stateless, all states are explicitly managed by upper-level Render Graph (not like NVRHI)
    // - sync problems.
    //   - upload->execution->upload: semaphore between upload and execution
    //   - family ownership transfer: semaphore between two cmds

    // VA_CommandListBufferManager:
    // - underlying command switch and management
    // - responsible for command pool keeping

    // RhiCommandList:
    // - states: idle -(acquire ctx)-> active -(submit ctx)-> idle
    // - submit with RhiTaskSubmission (explicit controlled by task system)

    enum class EVA_CommandListNativeState
    {
        Undefined,
        ReadyToBegin,
        Recording,
        ReadyToSubmit,
        Submitted
    };

    struct VA_CommandListNativeInternal;
    class IFRIT_VKRHI2_API VA_CommandListNative
    {
    public:
        VA_CommandListNative(VkCommandBuffer cmd);
        ~VA_CommandListNative();

        VkCommandBuffer            GetCmd();
        EVA_CommandListNativeState GetState();

        void                       Begin();
        void                       End();

    private:
        VA_CommandListNativeInternal* mInternal;
    };

    struct VA_CommandBufferManagerInternal;
    class IFRIT_VKRHI2_API VA_CommandBufferManager
    {
    public:
        VA_CommandBufferManager(VA_Device* device, ERhiPipelineType type, VA_Queue* queue);
        ~VA_CommandBufferManager();

        VA_CommandListNative* GetActiveCommandList();
        VA_CommandListNative* GetUploadCommandList();

        void                  SubmitActiveCommandList();
        void                  SubmitUploadCommandList();

    protected:
    private:
        VA_CommandBufferManagerInternal* mInternal;
    };

    struct VA_CommandListContextInternal;
    class IFRIT_VKRHI2_API VA_CommandListContext : public RHI::RhiCommandListContext
    {
    public:
        VA_CommandListContext(
            VA_Device* device, ERhiPipelineType type, VA_Queue* queue, VA_CommandListContext* primaryCmd);
        virtual ~VA_CommandListContext();

    private:
        VA_CommandListContextInternal* mInternal;
    };
} // namespace Ifrit::RHI::VulkanRHI2