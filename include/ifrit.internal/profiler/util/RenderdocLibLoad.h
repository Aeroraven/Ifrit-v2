#pragma once

#include "ifrit/profiler/ProfilerBase.h"

namespace Ifrit::Profiler::Internal::Renderdoc
{
    IFRIT_PROFILER_API bool LoadRenderdocLibrary();
    IFRIT_PROFILER_API void RequestRenderdocCaptureStart();
    IFRIT_PROFILER_API void RequestRenderdocCaptureEnd();
} // namespace Ifrit::Profiler::Internal::Renderdoc