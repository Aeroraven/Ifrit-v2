#include "ifrit/profiler/ProfilerSystem.h"
#include "ifrit/profiler/framecapture/FrameCapturer.h"
#include "ifrit/profiler/framecapture/RenderdocCapturer.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/application/ApplicationState.h"

namespace Ifrit::Profiler
{
    struct ProfilerSystemData
    {
        Owner<FrameCapturer> mFrameCapturer;
        bool                 mHasStartedCapture = false;
    };

    ProfilerSystem::ProfilerSystem() : mData(new ProfilerSystemData()) {}

    ProfilerSystem::~ProfilerSystem() { delete mData; }

    void ProfilerSystem::OnInitialize(Runtime::IApplication* app)
    {
        if (mConfig.mFrameCaptureType == EFrameCaptureType::Renderdoc)
        {
            mData->mFrameCapturer = MakeOwner<RenderdocCapturer>();
        }
        else
        {
            IF_LOG_ERROR(
                "ProfilerSystem", "Unsupported frame capture type: {}", static_cast<int>(mConfig.mFrameCaptureType));
            return;
        }
    }

    void ProfilerSystem::OnShutdown() {}

    void ProfilerSystem::OnFrameBegin()
    {
        auto app   = Runtime::GetActiveApplication();
        auto state = app->GetApplicationState();
        if (state->mProfilerRequestFrameCapture)
        {
            mData->mFrameCapturer->StartCapture();
            state->mProfilerRequestFrameCapture = false;
            mData->mHasStartedCapture           = true;
        }
    }

    void ProfilerSystem::OnFrameEnd()
    {
        if (mData->mHasStartedCapture)
        {
            mData->mFrameCapturer->StopCapture();
            mData->mHasStartedCapture = false;
        }
    }
    Owner<RHI::RhiTaskSubmission> ProfilerSystem::OnPreRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }
    Owner<RHI::RhiTaskSubmission> ProfilerSystem::OnPostRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }
    void ProfilerSystem::OnUpdate(Runtime::Scene* scene) {}

} // namespace Ifrit::Profiler