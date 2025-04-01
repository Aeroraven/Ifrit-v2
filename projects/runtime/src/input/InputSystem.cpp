#include "ifrit/runtime/input/InputSystem.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"

using namespace Ifrit;
using namespace Ifrit::Runtime;

static InputSystem* activeInputSystem = nullptr;

void                key_callback_glfw_input_system(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        activeInputSystem->UpdateKeyStatus(key, 1);
    }
}

namespace Ifrit::Runtime
{

    IFRIT_APIDECL void InputSystem::Init()
    {
        for (auto& key : m_keyStatus)
        {
            key.stat = 0;
        }
        using namespace Ifrit::Display::Window;
        activeInputSystem   = this;
        auto windowProvider = static_cast<GLFWWindowProvider*>(m_app->GetDisplay());
        auto windowHandle   = static_cast<GLFWwindow*>(windowProvider->GetGLFWWindow());
        windowProvider->RegisterKeyCallback(key_callback_glfw_input_system);
        iInfo("Input system initialized");
    }

    IFRIT_APIDECL void InputSystem::OnFrameUpdate()
    {
        for (auto& key : m_keyStatus)
        {
            if (key.stat == 1)
            {
                key.stat = 0;
            }
        }
    }

} // namespace Ifrit::Runtime