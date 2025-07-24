
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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

#include "ifrit/runtime/application/Application.h"
#include "ifrit/display/presentation/window/WindowSelector.h"
#include "ifrit/rhi/platform/RhiSelector.h"
#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"
#include "ifrit/core/hal/HalDisplay.h"
#include "ifrit/core/hal/HalWindow.h"
namespace Ifrit::Runtime
{
    struct ApplicationPrivateData
    {
        bool m_ConsoleWindowVisible = true;
    };

    IFRIT_APIDECL      Application::Application() { m_Data = new ApplicationPrivateData(); }
    IFRIT_APIDECL      Application::~Application() { delete m_Data; }

    IFRIT_APIDECL void Application::Run(const ProjectProperty& info)
    {
        m_info = info;
        Start();
        m_windowProvider->Loop([this](int* unused) { Update(); });
        End();
    }

    IFRIT_APIDECL void Application::Start()
    {
        //
        auto dpiScaler = 1.0f;
        if (m_info.m_EnableDPIScaling)
        {
            dpiScaler = HAL::GetDisplayScale();
            IF_LOG_INFO("Application", "DPI scaling enabled: {:.2f}", dpiScaler);
        }

        // Setup Window
        Display::Window::WindowProviderSetupArgs winArgs;
        winArgs.useVulkan = (m_info.m_rhiType == AppRhiType::Vulkan);
        Display::Window::WindowSelector     selector;

        Display::Window::WindowProviderType providerType;
        if (m_info.m_displayProvider == AppDisplayProvider::GLFW)
        {
            providerType = Display::Window::WindowProviderType::GLFW;
        }
        m_windowProvider = selector.CreateWindowProvider(providerType, winArgs);
        m_windowProvider->Setup(
            static_cast<usize>(m_info.m_width * dpiScaler), static_cast<usize>(m_info.m_height * dpiScaler));

        // Setup RHI
        RHI::RhiInitializeArguments rhiArgs;
        rhiArgs.m_surfaceWidth                = static_cast<u32>(m_info.m_width * dpiScaler);
        rhiArgs.m_surfaceHeight               = static_cast<u32>(m_info.m_height * dpiScaler);
        rhiArgs.m_expectedComputeQueueCount   = m_info.m_rhiComputeQueueCount;
        rhiArgs.m_expectedGraphicsQueueCount  = m_info.m_rhiGraphicsQueueCount;
        rhiArgs.m_expectedTransferQueueCount  = m_info.m_rhiTransferQueueCount;
        rhiArgs.m_expectedSwapchainImageCount = m_info.m_rhiNumBackBuffers;
        rhiArgs.m_enableValidationLayer       = m_info.m_rhiDebugMode;
        if (!m_info.m_rhiDebugMode)
        {
            IF_LOG_WARNING("Application", "Debug mode is disabled, validation layers are not enabled");
        }
#ifdef _WIN32
        rhiArgs.m_win32.m_hInstance = GetModuleHandle(NULL);
        rhiArgs.m_win32.m_hWnd      = (HWND)m_windowProvider->GetWindowObject();
#endif
        if (m_info.m_rhiType == AppRhiType::Vulkan)
            rhiArgs.m_extensionGetter = [this](uint32_t* count) -> const char** {
                return m_windowProvider->GetVkRequiredInstanceExtensions(count);
            };

        RHI::RhiSelector    rhiSelector;
        RHI::RhiBackendType rhiType;
        switch (m_info.m_rhiType)
        {
            case AppRhiType::Vulkan:
                rhiType = RHI::RhiBackendType::Vulkan;
                break;
            default:
                IF_LOG_CRITICAL("Application", "Unsupported RHI type: {}", static_cast<int>(m_info.m_rhiType));
                return;
        }
        m_rhiLayer = rhiSelector.CreateBackend(rhiType, rhiArgs);

        // Setup RHI cache
        m_rhiLayer->SetCacheDirectory(m_info.m_cachePath);

        // Prepare shared render resource
        m_SharedRenderResource = MakeRef<SharedRenderResource>(m_rhiLayer.get());

        // Prepare internal shaders
        m_shaderRegistry = MakeRef<ShaderRegistry>(this);
        Internal::RegisterRuntimeInternalShaders(m_shaderRegistry.get());

        // Setup systems
        m_assetManager      = MakeRef<AssetManager>(m_info.m_assetPath, this);
        m_sceneAssetManager = MakeRef<SceneAssetManager>(m_info.m_scenePath, m_assetManager.get());
        // m_assetManager->LoadAssetDirectory();
        IF_LOG_INFO("Application", "Asset directory loaded from: {}", m_info.m_assetPath);

        m_sceneManager = MakeRef<SceneManager>(this);

        // Input System
        m_inputSystem = MakeRef<InputSystem>(this);

        // Timing Recorder
        m_timingRecorder = MakeRef<TimingRecorder>();

        // Renderer Wrapper
        m_RendererWrapper = MakeRef<RendererWrapper>(m_rhiLayer.get(), m_shaderRegistry.get(), GetProjectProperty());

        OnStart();
    }

    IFRIT_APIDECL void Application::Update()
    {
        m_timingRecorder->OnUpdate();
        m_sceneManager->InvokeActiveSceneUpdate();
        if (m_EnableRendererWrapper)
        {
            m_RendererWrapper->BeginFrame();
            for (auto& subsystem : m_Subsystems)
            {
                subsystem->OnFrameBegin();
            }
            for (auto& subsystem : m_Subsystems)
            {
                m_RendererWrapper->EnqueueGeneralTask(
                    [&](RHI::RhiTaskSubmission* prevSubmission) { return subsystem->OnPreRendering(prevSubmission); });
            }
        }
        OnUpdate();
        if (m_EnableRendererWrapper)
        {
            for (auto& subsystem : m_Subsystems)
            {
                subsystem->OnUpdate(m_sceneManager->GetActiveScene().get());
            }
            if (!m_ApplicationState.m_EditorMode)
            {
                m_RendererWrapper->DrawToScreen();
            }

            for (auto& subsystem : m_Subsystems)
            {
                m_RendererWrapper->EnqueueGeneralTask(
                    [&](RHI::RhiTaskSubmission* prevSubmission) { return subsystem->OnPostRendering(prevSubmission); });
            }
            m_RendererWrapper->EndFrame();
            for (auto& subsystem : m_Subsystems)
            {
                subsystem->OnFrameEnd();
            }
        }
        m_inputSystem->OnFrameUpdate();

        if (m_Data->m_ConsoleWindowVisible && m_ApplicationState.m_EditorMode)
        {
            HAL::HideConsoleWindow();
            m_Data->m_ConsoleWindowVisible = false;
        }
    }

    IFRIT_APIDECL void Application::End()
    {
        m_rhiLayer->WaitDeviceIdle();
        OnEnd();
        for (auto& subsystem : m_Subsystems)
        {
            subsystem->OnShutdown();
        }
    }

    IFRIT_APIDECL void Application::RegisterSubsystem(Owner<ISubsystem> subsystem)
    {
        auto ptr = subsystem.get();
        m_Subsystems.push_back(std::move(subsystem));
        ptr->OnInitialize(this);
    }

    IFRIT_APIDECL void Application::EnableRendererWrapper(bool enable) { m_EnableRendererWrapper = enable; }

    IFRIT_APIDECL RHI::RhiTexture* Application::GetDefaultColorImage() const
    {
        if (m_EnableRendererWrapper)
        {
            return m_RendererWrapper->GetDefaultColorImage().get();
        }
        else
        {
            return m_rhiLayer->GetSwapchainImage();
        }
    }

} // namespace Ifrit::Runtime