#include "ifrit/core/hal/HalWindow.h"

#ifdef _WIN32
    #include <windows.h>
    #include <winerror.h>
#endif

namespace Ifrit::HAL
{
    IFRIT_CORE_API void HideConsoleWindow()
    {
#ifdef _WIN32
        HWND consoleWindow = GetConsoleWindow();
        if (consoleWindow)
        {
            ShowWindow(consoleWindow, SW_HIDE);
        }
#endif
    }
} // namespace Ifrit::HAL