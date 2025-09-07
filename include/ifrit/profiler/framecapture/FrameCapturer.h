#pragma once
#include "ifrit/profiler/ProfilerBase.h"

namespace Ifrit::Profiler
{
    class IFRIT_PROFILER_API FrameCapturer
    {
    public:
        FrameCapturer()          = default;
        virtual ~FrameCapturer() = default;

        virtual void StartCapture() = 0;
        virtual void StopCapture()  = 0;
    };
} // namespace Ifrit::Profiler
