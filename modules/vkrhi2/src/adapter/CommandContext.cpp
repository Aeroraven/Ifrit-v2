#include "ifrit/vkrhi2/adapter/CommandContext.h"
#include "ifrit/vkrhi2/adapter/CommandBuffer.h"
#include "ifrit/vkrhi2/adapter/CommandSubmission.h"
#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/core/console/ConsoleObject.h"
#include "ifrit/vkrhi2/adapter/PipelineState.h"
#include "ifrit.internal/vkrhi2/adapter/CmdHelpersBarrier.h"

namespace Ifrit::RHI::VulkanRHI2
{
    static TConsoleVariable<bool> cvVulkanAsyncPipelineSetup(
        "cv.VulkanRHI2.CommandListTTL", true, "Vulkan Command List TTL", CVF_ReadOnly);

    struct VA_CommandListNativeInternal
    {
        VA_Device*                     mDevice;
        VA_Queue*                      mQueue;
        VA_CommandListPool*            mPool             = nullptr;
        VA_CommandListContext*         mImmediateContext = nullptr;

        Vec<Ref<VA_CommandTask>>       mCurrentSubTasks;
        EVA_CommandTaskState           mCurrentState = EVA_CommandTaskState::Invalid;

        Vec<Ref<VA_CommandSubmission>> mExternalToWait;
        Ref<VA_CommandSubmission>      mLastUploadTask;
        VkFence                        mExternalFence = VK_NULL_HANDLE;
        VkSemaphore                    mExternalSema  = VK_NULL_HANDLE;

        // Pipeline States
        RhiComputePipelineStateDesc    mCurrentComputePSO;
        RhiGraphicsPipelineStateDesc   mCurrentGraphicsPSO;
        Task::TaskReference            mComputePSOCompilation  = nullptr;
        Task::TaskReference            mGraphicsPSOCompilation = nullptr;
    };

    IFRIT_VKRHI2_API VA_CommandListContext::VA_CommandListContext(
        VA_Device* device, VA_Queue* queue, VA_CommandListContext* immediateContext)
    {
        mInternal                    = new VA_CommandListNativeInternal();
        mInternal->mDevice           = device;
        mInternal->mQueue            = queue;
        mInternal->mPool             = queue->AcquireCommandPool();
        mInternal->mImmediateContext = immediateContext;
    };

    IFRIT_VKRHI2_API VA_CommandListContext::~VA_CommandListContext()
    {
        if (mInternal->mPool && mInternal->mQueue)
        {
            mInternal->mQueue->ReleaseCommandPool(mInternal->mPool);
            mInternal->mPool = nullptr;
        }
        delete mInternal;
        mInternal = nullptr;
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::SetLastUploadingTask(Ref<RhiTaskSubmission> uploadTask)
    {
        mInternal->mLastUploadTask = std::static_pointer_cast<VA_CommandSubmission>(uploadTask);
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::AddCompletionCallback(Fn<void()> callback)
    {
        auto subTask = GetTaskSection(EVA_CommandTaskState::Execute);
        subTask->mCompletionCallbacks.push_back(callback);
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::NewTaskSection()
    {
        EndTaskSection();
        mInternal->mCurrentState = EVA_CommandTaskState::Wait;
        mInternal->mCurrentSubTasks.push_back(MakeRef<VA_CommandTask>());
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::EndTaskSection()
    {
        if (mInternal->mCurrentSubTasks.size())
        {
            auto& lastTask = mInternal->mCurrentSubTasks.back();
            IF_LOG_ASSERTION(
                "VA_CommandListContext", lastTask->mState == mInternal->mCurrentState, "Task state corrupted");
            IF_LOG_ASSERTION("VA_CommandListContext", lastTask->mState == EVA_CommandTaskState::Execute,
                "Task not ended properly, missing EndTaskSection call?");
            if (lastTask->mState == EVA_CommandTaskState::Execute)
            {
                lastTask->mState = EVA_CommandTaskState::Recorded;
            }
            lastTask->mCmd->End();
            mInternal->mCurrentState = EVA_CommandTaskState::Recorded;
        }
    }

    IFRIT_VKRHI2_API VA_CommandTask* VA_CommandListContext::GetTaskSection(EVA_CommandTaskState desiredState)
    {
        if (mInternal->mCurrentSubTasks.empty() || mInternal->mCurrentState > desiredState)
        {
            NewTaskSection();
        }
        auto& lastTask           = mInternal->mCurrentSubTasks.back();
        lastTask->mState         = desiredState;
        mInternal->mCurrentState = desiredState;
        return lastTask.get();
    }

    IFRIT_VKRHI2_API VA_CommandListNative* VA_CommandListContext::GetCommandBuffer()
    {
        auto subTask = GetTaskSection(EVA_CommandTaskState::Execute);
        if (subTask->mCmd == nullptr)
        {
            auto pool    = mInternal->mPool;
            auto poolCmd = pool->AllocateCommandBuffer();
            poolCmd->Begin();
            subTask->mCmd = poolCmd;
        }
        return subTask->mCmd;
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::RegisterDependencies(Vec<Ref<VA_CommandSubmission>> toWait)
    {
        for (auto& sub : toWait)
        {
            mInternal->mExternalToWait.push_back(sub);
        }
    }
    IFRIT_VKRHI2_API void VA_CommandListContext::RegisterExternalDependencies(VkFence extFence, VkSemaphore extSema)
    {
        mInternal->mExternalFence = extFence;
        mInternal->mExternalSema  = extSema;
    }

    IFRIT_VKRHI2_API Ref<VA_CommandSubmission> VA_CommandListContext::FlushAllTaskSections()
    {
        Ref<VA_CommandSubmission> ret   = nullptr;
        auto                      queue = mInternal->mQueue;
        EndTaskSection();
        if (mInternal->mCurrentSubTasks.size() == 0)
        {
            IF_LOG_WARNING("VA_CommandListContext", "No tasks to flush");
            return ret;
        }

        mInternal->mCurrentSubTasks[0]->mToWait                = mInternal->mExternalToWait;
        mInternal->mCurrentSubTasks.back()->mExternalFence     = mInternal->mExternalFence;
        mInternal->mCurrentSubTasks.back()->mExternalSemaphore = mInternal->mExternalSema;

        if (mInternal->mLastUploadTask)
        {
            mInternal->mCurrentSubTasks[0]->mToWait.push_back(mInternal->mLastUploadTask);
            mInternal->mLastUploadTask = nullptr;
        }

        mInternal->mExternalToWait.clear();
        mInternal->mExternalFence = VK_NULL_HANDLE;
        mInternal->mExternalSema  = VK_NULL_HANDLE;

        for (int i = 0; i < mInternal->mCurrentSubTasks.size(); i++)
        {
            auto& subTask = mInternal->mCurrentSubTasks[i];
            if (i != 0)
            {
                subTask->mToWait.push_back(mInternal->mCurrentSubTasks[i - 1]->mToSignal);
            }
            queue->EnqueueCommandTask(subTask);
        }
        ret = mInternal->mCurrentSubTasks.back()->mToSignal;
        mInternal->mCurrentSubTasks.clear();
        mInternal->mCurrentState = EVA_CommandTaskState::Invalid;
        return ret;
    }

    // Public API
    IFRIT_VKRHI2_API Ref<RhiTaskSubmission> VA_CommandListContext::FlushCommands(ERhiCommandSubmissionAction action)
    {
        auto ret                   = FlushAllTaskSections();
        mInternal->mLastUploadTask = nullptr;
        if (action == ERhiCommandSubmissionAction::CPUWaitForSubmission)
        {
            auto waitTask = VA_CommandTask::CreateCpuWaitTask();
            auto queue    = mInternal->mQueue;
            queue->EnqueueCommandTask(waitTask);
            waitTask->Wait();
        }
        return ret;
    }

    // Commands
    IFRIT_VKRHI2_API void VA_CommandListContext::CmdSetComputePipelineState(const RhiComputePipelineStateDesc& desc)
    {
        mInternal->mCurrentComputePSO = desc;
        auto psoCache                 = static_cast<VA_Device*>(mInternal->mDevice)->GetPipelineStateCache();
        auto taskScheduler            = Task::GetTaskScheduler();
        taskScheduler->EnqueueTask(
            [this, psoCache, desc](Task::Task* task, void* payload) {
                mInternal->mComputePSOCompilation = nullptr;
                auto pso                          = psoCache->GetComputePipeline(desc);
            },
            Task::ENamedTaskThread::AnyThread, {}, nullptr);
    }
    IFRIT_VKRHI2_API void VA_CommandListContext::CmdSetGraphicsPipelineState(const RhiGraphicsPipelineStateDesc& desc)
    {
        mInternal->mCurrentGraphicsPSO = desc;
        auto psoCache                  = static_cast<VA_Device*>(mInternal->mDevice)->GetPipelineStateCache();
        auto taskScheduler             = Task::GetTaskScheduler();
        taskScheduler->EnqueueTask(
            [this, psoCache, desc](Task::Task* task, void* payload) {
                mInternal->mGraphicsPSOCompilation = nullptr;
                auto pso                           = psoCache->GetGraphicsPipeline(desc);
            },
            Task::ENamedTaskThread::AnyThread, {}, nullptr);
    }
    IFRIT_VKRHI2_API void VA_CommandListContext::CmdBeginTransition(RhiTransition& transition)
    {
        if (transition.mTransitions.empty())
            return;
        if (transition.mState != ERhiTransitionState::Pending)
        {
            IF_LOG_CRITICAL("VA_CommandListContext", "Transition already begun or ended");
        }
        bool                requireSplitCmdBuf = (transition.mPipelineDst != transition.mPipelineSrc);

        VA_PipelineBarriers barriers;
        barriers.SetQueueInfo(mInternal->mDevice->GetActiveQueueFamilies());
        barriers.TranslateFromRhiBarriers(transition, true);
        auto cmd = GetCommandBuffer();
        barriers.ExecuteNative(cmd->GetCmd());

        if (requireSplitCmdBuf)
        {
            auto submission                      = FlushAllTaskSections();
            transition.mTransitionBeginSemaphore = submission;

            RegisterDependencies({ submission });
        }
        transition.mState = ERhiTransitionState::Begin;
    }
    IFRIT_VKRHI2_API void VA_CommandListContext::CmdBeginTransitionList(const Vec<Ref<RhiTransition>>& transitions)
    {

        bool shouldFlushCmds = false;
        for (const auto& transition : transitions)
        {
            if (transition->mTransitions.empty())
                continue;
            if (transition->mState != ERhiTransitionState::Pending)
            {
                IF_LOG_CRITICAL("VA_CommandListContext", "Transition already begun or ended");
            }
            bool                requireSplitCmdBuf = (transition->mPipelineDst != transition->mPipelineSrc);

            VA_PipelineBarriers barriers;
            barriers.SetQueueInfo(mInternal->mDevice->GetActiveQueueFamilies());
            barriers.TranslateFromRhiBarriers(*transition, true);
            auto cmd = GetCommandBuffer();
            barriers.ExecuteNative(cmd->GetCmd());

            shouldFlushCmds |= requireSplitCmdBuf;
        }
        if (shouldFlushCmds)
        {
            auto submission = FlushAllTaskSections();
            RegisterDependencies({ submission });
            for (const auto& transition : transitions)
            {
                if (transition->mTransitions.empty())
                    continue;
                bool requireSplitCmdBuf = (transition->mPipelineDst != transition->mPipelineSrc);
                if (requireSplitCmdBuf)
                {
                    transition->mTransitionBeginSemaphore = submission;
                }
                transition->mState = ERhiTransitionState::Begin;
            }
        }
        for (const auto& transition : transitions)
        {
            if (transition->mTransitions.empty())
                continue;
            transition->mState = ERhiTransitionState::Begin;
        }
    }

    IFRIT_VKRHI2_API void VA_CommandListContext::CmdEndTransition(RhiTransition& transition)
    {

        if (transition.mTransitions.empty())
            return;
        if (transition.mState != ERhiTransitionState::Begin)
        {
            IF_LOG_CRITICAL("VA_CommandListContext", "Transition not begun or already ended");
        }
        if (transition.mTransitions.empty())
            return;
        bool requireSplitCmdBuf = (transition.mPipelineDst != transition.mPipelineSrc);

        if (!requireSplitCmdBuf)
            return;

        auto                submissionCurrent = FlushAllTaskSections();

        VA_PipelineBarriers barriers;
        barriers.SetQueueInfo(mInternal->mDevice->GetActiveQueueFamilies());
        barriers.TranslateFromRhiBarriers(transition, false);
        auto cmd = GetCommandBuffer();
        barriers.ExecuteNative(cmd->GetCmd());

        auto submissionNative = CheckedPointerCast<VA_CommandSubmission>(transition.mTransitionBeginSemaphore);
        if (submissionNative)
        {
            RegisterDependencies({ submissionNative, submissionCurrent });
        }
        transition.mState = ERhiTransitionState::End;
    }
    IFRIT_VKRHI2_API void VA_CommandListContext::CmdEndTransitionList(const Vec<Ref<RhiTransition>>& transitions)
    {
        bool                           shouldFlushCmds = false;

        Vec<Ref<VA_CommandSubmission>> submissionsToRegister;
        for (const auto& transition : transitions)
        {
            if (transition->mTransitions.empty())
                continue;
            if (transition->mState != ERhiTransitionState::Begin)
            {
                IF_LOG_CRITICAL("VA_CommandListContext", "Transition not begun or already ended");
            }
            bool requireSplitCmdBuf = (transition.get()->mPipelineDst != transition->mPipelineSrc);

            if (!requireSplitCmdBuf)
                continue;

            shouldFlushCmds |= requireSplitCmdBuf;
        }
        if (shouldFlushCmds)
        {
            auto submissionCurrent = FlushAllTaskSections();
            submissionsToRegister.push_back(submissionCurrent);
            for (const auto& transition : transitions)
            {
                if (transition->mTransitions.empty())
                    continue;
                bool requireSplitCmdBuf = (transition->mPipelineDst != transition->mPipelineSrc);
                if (!requireSplitCmdBuf)
                    continue;

                VA_PipelineBarriers barriers;
                barriers.SetQueueInfo(mInternal->mDevice->GetActiveQueueFamilies());
                barriers.TranslateFromRhiBarriers(*transition, false);
                auto cmd = GetCommandBuffer();
                barriers.ExecuteNative(cmd->GetCmd());

                auto submissionNative = CheckedPointerCast<VA_CommandSubmission>(transition->mTransitionBeginSemaphore);
                if (submissionNative)
                {
                    submissionsToRegister.push_back(submissionNative);
                }
                transition->mState = ERhiTransitionState::End;
            }
            RegisterDependencies(submissionsToRegister);
        }
    }
} // namespace Ifrit::RHI::VulkanRHI2