#pragma once
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/scene/SceneAssetManager.h"
#include "ifrit/display/presentation/window/WindowProvider.h"
#include <string>

namespace Ifrit::Core {
enum class ApplicationRhiType { Vulkan, DX12, OpenGL, Software };
enum class ApplicationDisplayProvider { GLFW };

struct ApplicationCreateInfo {
  std::string m_name;
  std::string m_version;
  ApplicationRhiType m_rhiType = ApplicationRhiType::Vulkan;
  ApplicationDisplayProvider m_displayProvider =
      ApplicationDisplayProvider::GLFW;
  uint32_t m_width = 1980;
  uint32_t m_height = 1080;
  std::string m_assetPath;
  std::string m_scenePath;

  uint32_t m_rhiGraphicsQueueCount = 1;
  uint32_t m_rhiTransferQueueCount = 1;
  uint32_t m_rhiComputeQueueCount = 1;
  uint32_t m_rhiNumBackBuffers = 2;
};
class IFRIT_APIDECL Application : public IApplication {
protected:
  std::shared_ptr<AssetManager> m_assetManager;
  std::shared_ptr<SceneAssetManager> m_sceneAssetManager;
  std::unique_ptr<GraphicsBackend::Rhi::RhiBackend> m_rhiLayer;
  std::unique_ptr<Display::Window::WindowProvider> m_windowProvider;
  ApplicationCreateInfo m_info;

private:
  void start();
  void update();
  void end();
  inline bool applicationShouldClose() { return true; }

public:
  virtual void onStart() override {}
  virtual void onUpdate() override {}
  virtual void onEnd() override {}
  void run(const ApplicationCreateInfo &info);

  inline virtual Ifrit::GraphicsBackend::Rhi::RhiBackend *
  getRhiLayer() override {
    return m_rhiLayer.get();
  }
};
} // namespace Ifrit::Core