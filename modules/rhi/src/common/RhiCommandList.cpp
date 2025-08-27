#include "ifrit/rhi/common/RhiCommandList.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/rhi/common/RhiDeviceProcs.h"

namespace Ifrit::RHI
{
    IFRIT_APIDECL void RhiCommandListBase::Submit()
    {
        Finalize();
        AcquireActiveContext();
    }

    IFRIT_APIDECL void RhiCommandListBase::Finalize()
    {
        if (mUploadContext)
        {
            SubmitUploadContext();
        }
        if (mActiveContext)
        {
            SubmitActiveContext();
        }
    }

    IFRIT_APIDECL void RhiCommandListBase::Init() { AcquireActiveContext(); }

    IFRIT_APIDECL void RhiCommandListBase::SwitchPipeline(ERhiCommandListPipelineType type)
    {
        // TODO
    }

    IFRIT_APIDECL RhiCommandListContext* RhiCommandListBase::GetActiveContext()
    {
        if (mUploadContext)
        {
            SubmitUploadContext();
        }
        return InternalGetActiveContext();
    }

    IFRIT_APIDECL RhiCommandListContext* RhiCommandListBase::GetUploadContext()
    {
        auto RHIProcs = mContext->GetDeviceRHIFunctions();
        if (!mUploadContext)
        {
            AcquireUploadContext();
        }
        return InternalGetUploadContext();
    }

    IFRIT_APIDECL void RhiCommandListBase::AcquireActiveContext()
    {
        IF_LOG_ASSERTION(
            "RhiCommandListBase", mRhiPipeline != ERhiCommandListPipelineType::Invalid, "Pipeline not set");
        IF_LOG_ASSERTION("RhiCommandListBase", !mActiveContext, "Active context already acquired");

        auto RHIProcs = mContext->GetDeviceRHIFunctions();
        if (!mActiveContext)
        {
            mActiveContext = RHIProcs->AcquireCommandListContext(mRhiPipeline, mImmediateCmdList);
        }
    }

    IFRIT_APIDECL void RhiCommandListBase::AcquireUploadContext()
    {
        IF_LOG_ASSERTION("RhiCommandListBase", !mUploadContext, "Upload context already acquired");

        auto RHIProcs = mContext->GetDeviceRHIFunctions();
        if (!mUploadContext)
        {
            mUploadContext =
                RHIProcs->AcquireCommandListContext(ERhiCommandListPipelineType::Graphics, mImmediateCmdList);
        }
    }

    IFRIT_APIDECL void RhiCommandListBase::SubmitActiveContext()
    {
        IF_LOG_ASSERTION("RhiCommandListBase", mActiveContext != nullptr, "No active context to submit");
        IF_LOG_ASSERTION("RhiCommandListBase", mUploadContext == nullptr, "Upload context must be submitted first");

        auto RHIProcs = mContext->GetDeviceRHIFunctions();
        if (mActiveContext)
        {
            Vec<Ref<RhiTaskSubmission>> toWait = mWaitSubmissions;
            if (mLastUploadContextSubmission)
            {
                toWait.push_back(mLastUploadContextSubmission);
            }
            for (auto& sub : mWaitSubmissions)
            {
                toWait.push_back(sub);
            }
            auto signaled = RHIProcs->SubmitCommandListContext(mActiveContext.get(), mImmediateCmdList, toWait);
            mLastActiveContextSubmission = signaled;
            mWaitSubmissions.clear();
            if (!IsImmediate())
            {
                RHIProcs->ReleaseCommandListContext(std::move(mActiveContext), mImmediateCmdList);
            }
        }
    }

    IFRIT_APIDECL void RhiCommandListBase::SubmitUploadContext()
    {
        IF_LOG_ASSERTION("RhiCommandListBase", mUploadContext != nullptr, "No upload context to submit");
        IF_LOG_ASSERTION("RhiCommandListBase", mActiveContext == nullptr, "Active context must be submitted first");

        auto RHIProcs = mContext->GetDeviceRHIFunctions();
        if (mUploadContext)
        {
            Vec<Ref<RhiTaskSubmission>> toWait = mWaitSubmissions;
            for (auto& sub : mWaitSubmissions)
            {
                toWait.push_back(sub);
            }
            auto signaled = RHIProcs->SubmitCommandListContext(mUploadContext.get(), mImmediateCmdList, toWait);
            mLastUploadContextSubmission = signaled;
            mWaitSubmissions.clear();

            RHIProcs->ReleaseCommandListContext(std::move(mUploadContext), mImmediateCmdList);
        }
    }

} // namespace Ifrit::RHI