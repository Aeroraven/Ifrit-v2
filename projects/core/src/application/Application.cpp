#include "ifrit/core/application/Application.h"
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
  rhiArgs.m_enableValidationLayer = true;
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
  case ApplicationRhiType::Software:
    rhiType = GraphicsBackend::Rhi::RhiBackendType::Software;
    break;
  default:
    throw std::runtime_error("RHI not supported");
  }
  m_rhiLayer = rhiSelector.createBackend(rhiType, rhiArgs);

  // Setup systems
  m_assetManager = std::make_shared<AssetManager>(m_info.m_assetPath, this);
  m_sceneAssetManager = std::make_shared<SceneAssetManager>(
      m_info.m_scenePath, m_assetManager.get());
  m_assetManager->loadAssetDirectory();

  onStart();
}

IFRIT_APIDECL void Application::update() { onUpdate(); }

IFRIT_APIDECL void Application::end() {
  m_rhiLayer->waitDeviceIdle();
  onEnd();
}

} // namespace Ifrit::Core