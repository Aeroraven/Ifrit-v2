#include "ifrit/profiler/framecapture/RenderdocCapturer.h"
#include "ifrit.internal/profiler/util/RenderdocLibLoad.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Profiler
{
    RenderdocCapturer::RenderdocCapturer()
    {
        if (!Internal::Renderdoc::LoadRenderdocLibrary())
        {
            IF_LOG_ERROR("Renderdoc", "Failed to load Renderdoc library.");
        }
    }
    void RenderdocCapturer::StartCapture() { Internal::Renderdoc::RequestRenderdocCaptureStart(); }
    void RenderdocCapturer::StopCapture() { Internal::Renderdoc::RequestRenderdocCaptureEnd(); }
} // namespace Ifrit::Profiler