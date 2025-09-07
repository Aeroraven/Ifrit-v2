#pragma once
#include "ifrit/core/base/IfritBase.h"

namespace Ifrit::Profiler
{
    enum class EFrameCaptureType : u8
    {
        Renderdoc
    };

    struct ProfilerConfig
    {
        EFrameCaptureType mFrameCaptureType = EFrameCaptureType::Renderdoc;
    };

} // namespace Ifrit::Profiler