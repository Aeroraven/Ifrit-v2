
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
#include "ifrit/core/logging/Logging.h"
#include "ifrit/runtime/application/ApplicationState.h"
using namespace Ifrit;
using namespace Ifrit::Runtime;

static InputSystem* activeInputSystem = nullptr;

void                InputSystemKeyCallbackGlfw(int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT)
    {
        activeInputSystem->UpdateKeyStatus(key, 1);
    }
}

void InputSystemMousePositionCallbackGlfw(double x, double y)
{
    auto appState = Runtime::GetActiveApplication()->GetApplicationState();
    if (appState->m_EditorMode)
    {
        x = (x - appState->mEditorViewportX) / appState->mEditorViewportWidth;
        y = (y - appState->mEditorViewportY) / appState->mEditorViewportHeight;
    }
    else
    {
        auto projectProperty = Runtime::GetActiveApplication()->GetProjectProperty();
        x                    = (x / projectProperty.m_width);
        y                    = (y / projectProperty.m_height);
    }
    activeInputSystem->UpdateMousePosition(static_cast<float>(x), static_cast<float>(y));
}

void InputSystemMouseButtonCallbackGlfw(int button, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        activeInputSystem->UpdateMouseButtonStatus(button, 1);
    }
    else if (action == GLFW_RELEASE)
    {
        activeInputSystem->UpdateMouseButtonStatus(button, 0);
    }
}

namespace Ifrit::Runtime
{
    IFRIT_APIDECL      InputSystem::~InputSystem() {}

    IFRIT_APIDECL bool InputSystem::IsKeyPressed(EInputKeyCode key)
    {
        return m_keyStatus[static_cast<int>(key)].stat == 1;
    }
    IFRIT_APIDECL bool InputSystem::IsKeyReleased(EInputKeyCode key)
    {
        return m_keyStatus[static_cast<int>(key)].stat == 0;
    }
    IFRIT_APIDECL bool InputSystem::IsMouseButtonPressed(EInputMouseButton button)
    {
        return mMouseButtonStatus[static_cast<int>(button)].stat == 1;
    }
    IFRIT_APIDECL bool InputSystem::IsMouseButtonReleased(EInputMouseButton button)
    {
        return mMouseButtonStatus[static_cast<int>(button)].stat == 0;
    }
    IFRIT_APIDECL float InputSystem::GetMouseX() const { return mMouseX; }
    IFRIT_APIDECL float InputSystem::GetMouseY() const { return mMouseY; }

    IFRIT_APIDECL void  InputSystem::UpdateKeyStatus(u32 key, u8 status) { m_keyStatus[key].stat = status; }
    IFRIT_APIDECL void  InputSystem::UpdateMousePosition(float x, float y)
    {
        mMouseX = x;
        mMouseY = y;
        // IF_LOG_INFO("Input", "Mouse Position Updated: ({}, {})", mMouseX, mMouseY);
    }
    IFRIT_APIDECL void InputSystem::UpdateMouseButtonStatus(u32 button, u8 status)
    {
        if (button < mMouseButtonStatus.size())
        {
            mMouseButtonStatus[button].stat = status;
            // IF_LOG_INFO("Input", "Mouse Button {} Status Updated: {}", button, status);
        }
        else
        {
            IF_LOG_ERROR("Input", "Invalid Mouse Button Index: {}", button);
        }
    }
    IFRIT_APIDECL void InputSystem::OnInitialize(IApplication* app)
    {
        m_app = app;
        for (auto& key : m_keyStatus)
        {
            key.stat = 0;
        }
        using namespace Ifrit::Display::Window;
        activeInputSystem   = this;
        auto windowProvider = static_cast<GLFWWindowProvider*>(m_app->GetDisplay());
        auto windowHandle   = static_cast<GLFWwindow*>(windowProvider->GetGLFWWindow());
        windowProvider->RegisterKeyCallback(InputSystemKeyCallbackGlfw);
        windowProvider->RegisterMousePostionCallback(InputSystemMousePositionCallbackGlfw);
        windowProvider->RegisterMouseButtonCallback(InputSystemMouseButtonCallbackGlfw);
        IF_LOG_INFO("InputSystem", "Input system initialized");
    }

    IFRIT_APIDECL void InputSystem::OnUpdate(Scene* scene)
    {
        for (auto& key : m_keyStatus)
        {
            if (key.stat == 1)
            {
                key.stat = 0;
            }
        }
    }
    IFRIT_APIDECL void InputSystem::OnShutdown() {}
    IFRIT_APIDECL void InputSystem::OnFrameBegin() {}
    IFRIT_APIDECL void InputSystem::OnFrameEnd() {}
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> InputSystem::OnPreRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr; // No pre-rendering tasks
    }
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> InputSystem::OnPostRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr; // No post-rendering tasks
    }
    IFRIT_APIDECL Owner<InputSystem> InputSystem::Create() { return MakeOwner<InputSystem>(); }

} // namespace Ifrit::Runtime