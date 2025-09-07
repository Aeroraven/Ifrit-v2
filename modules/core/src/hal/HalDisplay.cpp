#include "ifrit/core/hal/HalDisplay.h"
#include "ifrit/core/logging/Logging.h"
#ifdef _WIN32
    #include <windows.h>
    #include <winerror.h>
    #include <shellscalingapi.h>
    #pragma comment(lib, "Shcore.lib")
#endif

namespace Ifrit::HAL
{
    void SetDPIAwareness()
    {
#ifdef _WIN32
        // Try the newest DPI awareness API first (Windows 10 1703+)
        HMODULE user32 = GetModuleHandle("user32.dll");
        if (user32)
        {
            typedef BOOL(WINAPI * SetProcessDpiAwarenessContextProc)(DPI_AWARENESS_CONTEXT);
            SetProcessDpiAwarenessContextProc setProcessDpiAwarenessContext =
                (SetProcessDpiAwarenessContextProc)GetProcAddress(user32, "SetProcessDpiAwarenessContextW");

            if (setProcessDpiAwarenessContext)
            {
                if (setProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
                {
                    return;
                }
            }
        }

        // Fallback to older method (Windows 8.1+)
        if (SUCCEEDED(SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)))
        {
            return;
        }

        // Final fallback (Vista+)
        if (SetProcessDPIAware())
        {
            return;
        }

        IF_LOG_ERROR("HALDisplay", "Failed to set DPI awareness, using default settings");
#endif
    }

    IFRIT_APIDECL f32 GetDisplayScale()
    {
#ifdef _WIN32
        // Ensure DPI awareness is set
        static bool dpiAwarenessSet = false;
        if (!dpiAwarenessSet)
        {
            SetDPIAwareness();
            dpiAwarenessSet = true;
        }

        // Windows DPI scaling
        POINT    pt      = { 1, 1 };
        HMONITOR monitor = MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);

        UINT     dpiX, dpiY;
        if (SUCCEEDED(GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY)))
        {
            return static_cast<f32>(dpiX) / 96.0f;
        }

        // Fallback to older method
        HDC hdc = GetDC(NULL);
        if (hdc)
        {
            int dpi = GetDeviceCaps(hdc, LOGPIXELSX);
            ReleaseDC(NULL, hdc);
            return static_cast<f32>(dpi) / 96.0f;
        }

        IF_LOG_ERROR("HALDisplay", "Failed to get display scale, using default value of 1.0");
        return 1.0f;
#else
        static_assert(false, "GetDisplayScale not implemented for this platform");
        return 1.0f;
#endif
    }
} // namespace Ifrit::HAL