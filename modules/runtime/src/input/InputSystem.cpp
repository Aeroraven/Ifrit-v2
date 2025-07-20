
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "ifrit/runtime/input/InputSystem.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"
#include "ifrit/runtime/base/ApplicationInterface.h"

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
    IFRIT_APIDECL      InputSystem::InputSystem(IApplication* app) : m_app(app) { Init(); }
    IFRIT_APIDECL      InputSystem::~InputSystem() {}

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
        IF_LOG_INFO("InputSystem", "Input system initialized");
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