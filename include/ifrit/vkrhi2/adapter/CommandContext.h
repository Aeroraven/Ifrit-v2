#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/CommandSubmission.h"

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
    // submission, with orders w/ref to unreal
    // - stateless, all states are explicitly managed by upper-level Render Graph (not like NVRHI)
    // - sync problems.
    //   - family ownership transfer: semaphore between two cmds

    // VA_CommandListBufferManager:
    // - underlying command switch and management
    // - responsible for command pool keeping

    // RhiCommandList:
    // - states: idle -(acquire ctx)-> active -(submit ctx)-> idle
    // - submit with RhiTaskSubmission (explicit controlled by task system)

    // upload context (for transfer and initial) -> active context

    class VA_CommandListNative;

    struct VA_CommandListNativeInternal;
    class IFRIT_VKRHI2_API VA_CommandListContext : public IRhiCommandContext
    {
    public:
        VA_CommandListContext(VA_Device* device, VA_Queue* queue, VA_CommandListContext* immediateContext);
        virtual ~VA_CommandListContext();

        Ref<RhiTaskSubmission> FlushCommands(ERhiCommandSubmissionAction action);
        VA_CommandListNative*  GetCommandBuffer();
        void                   AddCompletionCallback(Fn<void()> callback) override;
        void                   SetLastUploadingTask(Ref<RhiTaskSubmission> uploadTask) override;

        void                   CmdSetComputePipelineState(const RhiComputePipelineStateDesc& desc) override;
        void                   CmdSetGraphicsPipelineState(const RhiGraphicsPipelineStateDesc& desc) override;
        void                   CmdSetShaderParameters(const RhiShaderParameter& params) override;

        void                   CmdBeginTransitionList(const Vec<Ref<RhiTransition>>& transitions) override;
        void                   CmdEndTransitionList(const Vec<Ref<RhiTransition>>& transitions) override;

        void                   CmdDispatch(u32 groupCountX, u32 groupCountY, u32 groupCountZ) override;

        void                   RegisterExternalDependencies(VkFence extFence, VkSemaphore extSema);
        void                   RegisterDependencies(Vec<Ref<VA_CommandSubmission>> toWait);

    private:
        void                      ApplyShaderParameterChange();
        void                      ApplyPipelineStateChange();

        void                      NewTaskSection();
        void                      EndTaskSection();
        Ref<VA_CommandSubmission> FlushAllTaskSections();
        VA_CommandTask*           GetTaskSection(EVA_CommandTaskState desiredState);

    private:
        VA_CommandListNativeInternal* mInternal;
    };

} // namespace Ifrit::RHI::VulkanRHI2