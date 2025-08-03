#pragma once

#include "ifrit/profiler/ProfilerBase.h"
#include "ifrit/profiler/framecapture/FrameCapturer.h"

namespace Ifrit::Profiler
{
    class IFRIT_PROFILER_API RenderdocCapturer : public FrameCapturer
    {
    public:
        RenderdocCapturer();
        ~RenderdocCapturer() override = default;

        void StartCapture() override;
        void StopCapture() override;
    };
} // namespace Ifrit::Profiler