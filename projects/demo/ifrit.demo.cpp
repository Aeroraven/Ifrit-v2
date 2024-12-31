
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

#ifndef IFRIT_DLL
#define IFRIT_DLL
#endif
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/Core.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"
#include <numbers>
#include <thread>

#define WINDOW_WIDTH 1980
#define WINDOW_HEIGHT 1080

using namespace Ifrit::GraphicsBackend::Rhi;
using namespace Ifrit::MeshProcLib::MeshProcess;
using namespace Ifrit::Math;
using namespace Ifrit::Core;
using namespace Ifrit::Common::Utility;

// Glfw key function here
float movLeft = 0, movRight = 0, movTop = 0, movBottom = 0, movFar = 0,
      movNear = 0, movRot = 0;

void key_callback(int key, int scancode, int action, int mods) {
  auto scale = 2.5f;
  if (key == GLFW_KEY_A && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movLeft += scale;

  if (key == GLFW_KEY_D && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movRight += scale;

  if (key == GLFW_KEY_W && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movTop += scale;

  if (key == GLFW_KEY_S && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movBottom += scale;

  if (key == GLFW_KEY_E && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movFar += scale;

  if (key == GLFW_KEY_F && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movNear += scale;

  if (key == GLFW_KEY_Z && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movRot += scale * 0.003f;

  if (key == GLFW_KEY_X && (action == GLFW_REPEAT || action == GLFW_PRESS))
    movRot -= scale * 0.003f;
}

class DemoApplication : public Ifrit::Core::Application {
private:
  RhiScissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
  std::shared_ptr<RhiRenderTargets> renderTargets;
  std::shared_ptr<RhiColorAttachment> colorAttachment;
  std::shared_ptr<RhiDepthStencilAttachment> depthAttachment;
  std::shared_ptr<SyaroRenderer> renderer;
  RhiTexture *swapchainImg;
  RendererConfig renderConfig;
  float timing = 0;

public:
  void onStart() override {
    renderer = std::make_shared<SyaroRenderer>(this);
    m_windowProvider->registerKeyCallback(key_callback);
    auto bistroObj =
        m_assetManager->getAssetByName<GLTFAsset>("Bistro/bistro.gltf");
    // Renderer config
    renderConfig.m_antiAliasingType = AntiAliasingType::TAA;
    renderConfig.m_shadowConfig.m_maxDistance = 6000.0f;

    // Scene
    auto s = m_sceneAssetManager->createScene("TestScene2");
    auto node = s->addSceneNode();

    auto cameraGameObject = node->addGameObject("camera");
    auto camera = cameraGameObject->addComponent<Camera>();
    camera->setCameraType(CameraType::Perspective);
    camera->setMainCamera(true);
    camera->setAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
    camera->setFov(60.0f / 180.0f * std::numbers::pi_v<float>);
    camera->setFar(6000.0f);
    camera->setNear(30.01f);

    auto cameraTransform = cameraGameObject->getComponent<Transform>();
    cameraTransform->setPosition({0.0f, 0.5f, -1.25f});
    cameraTransform->setRotation({0.0f, 0.1f, 0.0f});
    cameraTransform->setScale({1.0f, 1.0f, 1.0f});

    auto lightGameObject = node->addGameObject("sun");
    auto light = lightGameObject->addComponent<Light>();
    auto lightTransform = lightGameObject->getComponent<Transform>();
    // make light dir (0,-1,-1)=> eulerX=135deg
    lightTransform->setRotation({120.0 / 180.0f * std::numbers::pi_v<float>,
                                 -15.0 / 180.0f * std::numbers::pi_v<float>,
                                 0.0f});
    light->setShadowMap(true);
    light->setShadowMapResolution(2048);
    light->setAffectPbrSky(true);

    auto prefabs = bistroObj->getPrefabs();
    uint32_t numMeshes = 0;
    for (auto &prefab : prefabs) {
      numMeshes++;
      if (numMeshes < 910) {
        // continue;
      }
      node->addGameObjectTransferred(std::move(prefab->m_prefab));
    }

    // Render targets
    auto rt = m_rhiLayer.get();
    auto depthImage = rt->createDepthRenderTexture(WINDOW_WIDTH, WINDOW_HEIGHT);
    swapchainImg = rt->getSwapchainImage();
    renderTargets = rt->createRenderTargets();
    colorAttachment =
        rt->createRenderTarget(swapchainImg, {0.0f, 0.0f, 0.0f, 1.0f},
                               RhiRenderTargetLoadOp::Clear, 0, 0);
    depthAttachment = rt->createRenderTargetDepthStencil(
        depthImage, {{}, 1.0f}, RhiRenderTargetLoadOp::Clear);
    renderTargets->setColorAttachments({colorAttachment.get()});
    renderTargets->setDepthStencilAttachment(depthAttachment.get());
    renderTargets->setRenderArea(scissor);
  }

  void onUpdate() override {
    auto cameraGameObject = m_sceneAssetManager->getScene("TestScene2")
                                ->getRootNode()
                                ->getChildren()[0]
                                ->getGameObject(0);
    auto camera = cameraGameObject->getComponent<Transform>();
    timing = timing + 0.00f;
    camera->setPosition({0.0f + movRight - movLeft + 0.5f * std::sin(timing),
                         150.0f + movTop - movBottom,
                         200.25f + movFar - movNear});
    camera->setRotation({0.0f, movRot, 0.0f});
    auto sFrameStart = renderer->beginFrame();
    auto renderComplete = renderer->render(
        m_sceneAssetManager->getScene("TestScene2").get(), nullptr,
        renderTargets.get(), renderConfig, {sFrameStart.get()});
    renderer->endFrame({renderComplete.get()});

    // sleep for 50ms
    // std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  void onEnd() override {}
};

int main() {
  Ifrit::Core::ApplicationCreateInfo info;
  info.m_assetPath = IFRIT_DEMO_ASSET_PATH;
  info.m_scenePath = IFRIT_DEMO_SCENE_PATH;
  info.m_displayProvider = Ifrit::Core::ApplicationDisplayProvider::GLFW;
  info.m_rhiType = Ifrit::Core::ApplicationRhiType::Vulkan;
  info.m_width = 1980;
  info.m_height = 1080;
  info.m_rhiComputeQueueCount = 1;
  info.m_rhiGraphicsQueueCount = 1;
  info.m_rhiTransferQueueCount = 1;
  info.m_rhiNumBackBuffers = 2;
  info.m_name = "Ifrit-v2";
  info.m_cachePath = IFRIT_DEMO_CACHE_PATH;
  info.m_rhiDebugMode = true;

  DemoApplication app;
  app.run(info);
  return 0;
}