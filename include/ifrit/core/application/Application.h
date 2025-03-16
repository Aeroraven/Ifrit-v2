
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

#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/scene/SceneAssetManager.h"
#include "ifrit/core/scene/SceneManager.h"
#include "ifrit/display/presentation/window/WindowProvider.h"
#include <string>

namespace Ifrit::Core {
enum class ApplicationRhiType { Vulkan, DX12, OpenGL, Software };
enum class ApplicationDisplayProvider { GLFW };

struct ApplicationCreateInfo {
  std::string m_name;
  std::string m_version;
  std::string m_cachePath;
  ApplicationRhiType m_rhiType = ApplicationRhiType::Vulkan;
  ApplicationDisplayProvider m_displayProvider = ApplicationDisplayProvider::GLFW;
  u32 m_width = 1980;
  u32 m_height = 1080;
  std::string m_assetPath;
  std::string m_scenePath;

  u32 m_rhiGraphicsQueueCount = 1;
  u32 m_rhiTransferQueueCount = 1;
  u32 m_rhiComputeQueueCount = 1;
  u32 m_rhiNumBackBuffers = 2;
  u32 m_rhiDebugMode = 0;
};
class IFRIT_APIDECL Application : public IApplication {
protected:
  std::shared_ptr<AssetManager> m_assetManager;
  std::shared_ptr<SceneManager> m_sceneManager;
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

  inline virtual Ifrit::GraphicsBackend::Rhi::RhiBackend *getRhiLayer() override { return m_rhiLayer.get(); }

  inline std::string getCacheDirectory() const override { return m_info.m_cachePath; }
};
} // namespace Ifrit::Core