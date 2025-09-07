#pragma once

#include "RhiBaseTypes.h"
#include "RhiApi.h"
#include "ifrit/rhi/common/RhiCommand.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/rhi/common/RhiDevice.h"
#include "ifrit/rhi/common/RhiPipeline.h"
#include "ifrit/rhi/common/RhiTransition.h"

namespace Ifrit::RHI
{
    class RhiCommandListBase;
    class RhiCommandListExecutor;

    // ===== Command List Context =====


    enum class ERhiCommandSubmissionAction
    {
        None,
        CPUWaitForSubmission,
    };

    class IFRIT_RHI_API IRhiCommandContext
     {
     public:
        virtual ~IRhiCommandContext() = default;
 
        virtual void                   AddCompletionCallback(Fn<void()> callback)              = 0;
        virtual Ref<RhiTaskSubmission> FlushCommands(ERhiCommandSubmissionAction action)       = 0;
        virtual void                   SetLastUploadingTask(Ref<RhiTaskSubmission> uploadTask) = 0;

        virtual void                   CmdSetComputePipelineState(const RhiComputePipelineStateDesc& desc)   = 0;
        virtual void                   CmdSetGraphicsPipelineState(const RhiGraphicsPipelineStateDesc& desc) = 0;

        virtual void                   CmdBeginTransition(RhiTransition& transition)                      = 0;
        virtual void                   CmdEndTransition(RhiTransition& transition)                        = 0;
        virtual void                   CmdBeginTransitionList(const Vec<Ref<RhiTransition>>& transitions) = 0;
        virtual void                   CmdEndTransitionList(const Vec<Ref<RhiTransition>>& transitions)   = 0;
     };
 
     // ===== Command List Base (Recording) =====
    class IFRIT_RHI_API RhiCommandListBase
     {
     public:
        virtual ~RhiCommandListBase() = default;
 
        void                EnqueueLambda(Fn<void(RhiCommandListBase*)> func);
 
        void                SetComputePipelineState(const RhiComputePipelineStateDesc& desc);
        void                SetGraphicsPipelineState(const RhiGraphicsPipelineStateDesc& desc);
        void                BeginTransitions(const Vec<Ref<RhiTransition>>& transitions);
        void                EndTransitions(const Vec<Ref<RhiTransition>>& transitions);
 
        IRhiCommandContext* GetActiveContext();
        IRhiCommandContext* GetUploadContext();
        inline bool         IsImmediate() const { return mImmediateCmdList == nullptr; }
 
    protected:
        void EnqueueRHICommand(Owner<RhiCommand> cmd);
 
    protected:
        IRhiCommandContext*         mActiveContextImm = nullptr;
        Owner<IRhiCommandContext>   mActiveContext;
        Owner<IRhiCommandContext>   mUploadContext;
        Ref<RhiTaskSubmission>      mLastUploadTask;
 
        ERhiCommandListPipelineType mRhiPipeline = ERhiCommandListPipelineType::Invalid;
        Vec<Owner<RhiCommand>>      mCommands;
        Vec<Ref<RhiTaskSubmission>> mWaitSubmissions;
 
        RhiCommandListBase*         mImmediateCmdList = nullptr;
        bool                        mValid            = true;
 
         friend class RhiCommandListExecutor;
     };
 
    class IFRIT_RHI_API RhiCommandListImmediate : public RhiCommandListBase
    {
    public:
        RhiCommandListImmediate();
        virtual ~RhiCommandListImmediate() = default;

        void Initialize();
    };

     // ===== Command List Executor =====
    struct RhiCommandListExecutorInternal;
    class IFRIT_RHI_API RhiCommandListExecutor : public NonCopyable
     {
     public:
        RhiCommandListExecutor();
        ~RhiCommandListExecutor();

        RhiCommandListImmediate* GetImmediateCmdList();

        void                     Init();
        void                     PreFinalize();
 
     private:
        RhiCommandListExecutorInternal* mInternal;
     };

    IFRIT_RHI_API RhiCommandListExecutor* GetCommandListExecutor();
    IFRIT_RHI_API void                    UnloadCommandListExecutor();
 } // namespace Ifrit::RHI