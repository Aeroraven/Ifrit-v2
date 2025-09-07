 #include "ifrit/rhi/common/RhiCommandList.h"
 #include "ifrit/core/logging/Logging.h"
#include "ifrit/rhi/common/RhiDynamicUtils.h"
#include "ifrit/rhi/common/RhiInterface.h"
#include "ifrit/core/console/ConsoleObject.h"
 
 namespace Ifrit::RHI
 {
 
    static TConsoleVariable<u32> cvRHIEnableTranslationThread(
        "cv.RHI.EnableTranslationThread", false, "Enable RHI Translation Thread", CVF_ReadOnly);

    // ===== Command List Base (Recording) =====
    // IRhiCommandContext* GetActiveContext() const;
    // IRhiCommandContext* GetUploadContext() const;

    IFRIT_APIDECL IRhiCommandContext* RhiCommandListBase::GetActiveContext()
    {

        if (mUploadContext)
        {
            // IF_LOG_INFO("RhiCommandListBase", "Flushing upload context before getting active context");
            mUploadContext->SetLastUploadingTask(mLastUploadTask);
            auto uploadFlush = mUploadContext->FlushCommands(ERhiCommandSubmissionAction::None);
            mLastUploadTask  = uploadFlush;
        }

        if (IsImmediate())
        {
            if (mLastUploadTask)
                mActiveContextImm->SetLastUploadingTask(mLastUploadTask);
            return mActiveContextImm;
        }
        else
        {
            IF_LOG_ASSERTION("RhiCommandListBase", mActiveContext != nullptr, "Active context is null");
            return mActiveContext.get();
        }
    }

    IFRIT_APIDECL IRhiCommandContext* RhiCommandListBase::GetUploadContext()
    {
        if (!mUploadContext)
        {
            auto backend   = GetRhiBackend();
            mUploadContext = std::move(backend->GetUploadContext());
        }
        return mUploadContext.get();
    }

    IFRIT_APIDECL void RhiCommandListBase::EnqueueRHICommand(Owner<RhiCommand> cmd)
    {
        IF_LOG_ASSERTION("RhiCommandListBase", mValid, "Cannot enqueue command to invalid command list");
        if (!mValid)
            return;

        if (cvRHIEnableTranslationThread.GetValue())
        {
            mCommands.push_back(std::move(cmd));
        }
        else
        {
            cmd->Execute(this);
        }
    }

    IFRIT_APIDECL void RhiCommandListBase::EnqueueLambda(Fn<void(RhiCommandListBase*)> func)
    {
        EnqueueRHICommand(MakeOwner<RhiCmd_Lambda>(std::move(func)));
    }
    IFRIT_APIDECL void RhiCommandListBase::SetComputePipelineState(const RhiComputePipelineStateDesc& desc)
    {
        if (mRhiPipeline != ERhiCommandListPipelineType::Compute
            || mRhiPipeline == ERhiCommandListPipelineType::Graphics)
        {
            IF_LOG_ASSERTION(
                "RhiCommandListBase", false, "Cannot set compute pipeline state on non-compute command list");
            return;
        }

        EnqueueRHICommand(MakeOwner<RhiCmd_SetComputePipelineState>(desc));
    }
    IFRIT_APIDECL void RhiCommandListBase::SetGraphicsPipelineState(const RhiGraphicsPipelineStateDesc& desc)
    {
        if (mRhiPipeline != ERhiCommandListPipelineType::Graphics)
        {
            IF_LOG_ASSERTION(
                "RhiCommandListBase", false, "Cannot set graphics pipeline state on non-graphics command list");
            return;
        }

        EnqueueRHICommand(MakeOwner<RhiCmd_SetGraphicsPipelineState>(desc));
    }

    // ===== Immediate Command List =====
    IFRIT_APIDECL RhiCommandListImmediate::RhiCommandListImmediate()
    {
        mImmediateCmdList = nullptr;
        mValid            = true;
        mRhiPipeline      = ERhiCommandListPipelineType::Graphics;
    }

    IFRIT_APIDECL void RhiCommandListImmediate::Initialize()
    {
        mActiveContextImm = GetRhiBackend()->GetImmediateContext();
        IF_LOG_INFO("RhiCommandListImmediate", "Initialized immediate command list");
    }

    // ===== Command List Executor =====
    struct RhiCommandListExecutorInternal
    {
        Owner<RhiCommandListImmediate> mImmediateCmdList;
    };

    IFRIT_APIDECL RhiCommandListExecutor::RhiCommandListExecutor()
    {
        mInternal                    = new RhiCommandListExecutorInternal();
        mInternal->mImmediateCmdList = MakeOwner<RhiCommandListImmediate>();
    }

    IFRIT_APIDECL RhiCommandListExecutor::~RhiCommandListExecutor()
    {
        if (mInternal)
            delete mInternal;
        mInternal = nullptr;
    }

    IFRIT_APIDECL RhiCommandListImmediate* RhiCommandListExecutor::GetImmediateCmdList()
    {
        return mInternal->mImmediateCmdList.get();
    }
    IFRIT_APIDECL void                    RhiCommandListExecutor::Init() { mInternal->mImmediateCmdList->Initialize(); }
    IFRIT_APIDECL void                    RhiCommandListExecutor::PreFinalize() { this->~RhiCommandListExecutor(); }

    IFRIT_APIDECL RhiCommandListExecutor* GetCommandListExecutor()
    {
        static RhiCommandListExecutor executor;
        return &executor;
    }
    IFRIT_APIDECL void UnloadCommandListExecutor()
    {
        auto executor = GetCommandListExecutor();
        executor->PreFinalize();
    }

 } // namespace Ifrit::RHI