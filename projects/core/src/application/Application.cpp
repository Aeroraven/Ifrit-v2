
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

#include "ifrit/core/application/Application.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/display/presentation/window/WindowSelector.h"
#include "ifrit/rhi/platform/RhiSelector.h"

namespace Ifrit::Core {

IFRIT_APIDECL void Application::run(const ApplicationCreateInfo &info) {
  m_info = info;
  start();
  m_windowProvider->loop([this](int *unused) { update(); });
  end();
}

IFRIT_APIDECL void Application::start() {

  // Setup Window
  Display::Window::WindowProviderSetupArgs winArgs;
  winArgs.useVulkan = (m_info.m_rhiType == ApplicationRhiType::Vulkan);
  Display::Window::WindowSelector selector;

  Display::Window::WindowProviderType providerType;
  if (m_info.m_displayProvider == ApplicationDisplayProvider::GLFW) {
    providerType = Display::Window::WindowProviderType::GLFW;
  }
  m_windowProvider = selector.createWindowProvider(providerType, winArgs);
  m_windowProvider->setup(m_info.m_width, m_info.m_height);

  // Setup RHI
  GraphicsBackend::Rhi::RhiInitializeArguments rhiArgs;
  rhiArgs.m_surfaceWidth = m_info.m_width;
  rhiArgs.m_surfaceHeight = m_info.m_height;
  rhiArgs.m_expectedComputeQueueCount = m_info.m_rhiComputeQueueCount;
  rhiArgs.m_expectedGraphicsQueueCount = m_info.m_rhiGraphicsQueueCount;
  rhiArgs.m_expectedTransferQueueCount = m_info.m_rhiTransferQueueCount;
  rhiArgs.m_expectedSwapchainImageCount = m_info.m_rhiNumBackBuffers;
  rhiArgs.m_enableValidationLayer = m_info.m_rhiDebugMode;
  if (!m_info.m_rhiDebugMode) {
    iWarn("Debug mode is disabled, validation layers are not enabled");
  }
#ifdef _WIN32
  rhiArgs.m_win32.m_hInstance = GetModuleHandle(NULL);
  rhiArgs.m_win32.m_hWnd = (HWND)m_windowProvider->getWindowObject();
#endif
  if (m_info.m_rhiType == ApplicationRhiType::Vulkan)
    rhiArgs.m_extensionGetter = [this](uint32_t *count) -> const char ** {
      return m_windowProvider->getVkRequiredInstanceExtensions(count);
    };

  GraphicsBackend::Rhi::RhiSelector rhiSelector;
  GraphicsBackend::Rhi::RhiBackendType rhiType;
  switch (m_info.m_rhiType) {
  case ApplicationRhiType::Vulkan:
    rhiType = GraphicsBackend::Rhi::RhiBackendType::Vulkan;
    break;
  default:
    throw std::runtime_error("RHI not supported");
  }
  m_rhiLayer = rhiSelector.createBackend(rhiType, rhiArgs);

  // Setup RHI cache
  m_rhiLayer->setCacheDirectory(m_info.m_cachePath);

  // Setup systems
  m_assetManager = std::make_shared<AssetManager>(m_info.m_assetPath, this);
  m_sceneAssetManager = std::make_shared<SceneAssetManager>(m_info.m_scenePath, m_assetManager.get());
  m_assetManager->loadAssetDirectory();
  m_sceneManager = std::make_shared<SceneManager>(this);

  onStart();
}

IFRIT_APIDECL void Application::update() { onUpdate(); }

IFRIT_APIDECL void Application::end() {
  m_rhiLayer->waitDeviceIdle();
  onEnd();
}

} // namespace Ifrit::Core