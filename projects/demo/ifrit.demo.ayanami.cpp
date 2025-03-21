
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

#include "ifrit/common/logging/Logging.h"
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
float movLeft = 0, movRight = 0, movTop = 0, movBottom = 0, movFar = 0, movNear = 0, movRot = 0;

namespace Ifrit {
class DemoApplicationAyanami : public Ifrit::Core::Application {
private:
  RhiScissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
  std::shared_ptr<RhiRenderTargets> renderTargets;
  std::shared_ptr<RhiColorAttachment> colorAttachment;
  std::shared_ptr<RhiTexture> depthImage;
  std::shared_ptr<RhiDepthStencilAttachment> depthAttachment;
  std::shared_ptr<AyanamiRenderer> renderer;
  RhiTexture *swapchainImg;
  RendererConfig renderConfig;
  float timing = 0;

public:
  void onStart() override {
    iInfo("DemoApplication::onStart()");
    renderer = std::make_shared<AyanamiRenderer>(this);
    auto bistroObj = m_assetManager->getAssetByName<GLTFAsset>("Fox/scene.gltf");
    // Scene
    auto s = m_sceneAssetManager->createScene("TestScene2");
    auto node = s->addSceneNode();

    auto cameraGameObject = node->addGameObject("camera");
    auto camera = cameraGameObject->addComponent<Camera>();
    camera->setCameraType(CameraType::Perspective);
    camera->setMainCamera(true);
    camera->setAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
    camera->setFov(60.0f / 180.0f * std::numbers::pi_v<float>);
    camera->setFar(200.0f);
    camera->setNear(1.00f);

    auto cameraTransform = cameraGameObject->getComponent<Transform>();
    cameraTransform->setPosition({0.0f, 0.5f, -1.25f});
    cameraTransform->setRotation({0.0f, 0.1f, 0.0f});
    cameraTransform->setScale({1.0f, 1.0f, 1.0f});

    auto lightGameObject = node->addGameObject("sun");
    auto light = lightGameObject->addComponent<Light>();
    auto lightTransform = lightGameObject->getComponent<Transform>();
    lightTransform->setRotation({120.0 / 180.0f * std::numbers::pi_v<float>, 0.0f, 0.0f});
    light->setShadowMap(true);
    light->setShadowMapResolution(2048);
    light->setAffectPbrSky(true);

    auto meshes = bistroObj->getPrefabs();
    for (auto &m : meshes) {
      auto t = m->m_prefab;
      auto meshDF = t->addComponent<Ayanami::AyanamiMeshDF>();
      meshDF->buildMeshDF(getCacheDirectory());
      auto transform = t->getComponent<Transform>();
      auto mat = transform->getModelToWorldMatrix();
      node->addGameObjectTransferred(std::move(m->m_prefab));
    }

    // Render targets
    auto rt = m_rhiLayer.get();
    depthImage = rt->createDepthTexture(WINDOW_WIDTH, WINDOW_HEIGHT);
    swapchainImg = rt->getSwapchainImage();
    renderTargets = rt->createRenderTargets();
    colorAttachment =
        rt->createRenderTarget(swapchainImg, {0.0f, 0.0f, 0.0f, 1.0f}, RhiRenderTargetLoadOp::Clear, 0, 0);
    depthAttachment = rt->createRenderTargetDepthStencil(depthImage.get(), {{}, 1.0f}, RhiRenderTargetLoadOp::Clear);
    renderTargets->setColorAttachments({colorAttachment.get()});
    renderTargets->setDepthStencilAttachment(depthAttachment.get());
    renderTargets->setRenderArea(scissor);

    iInfo("Done");
  }

  void onUpdate() override {
    auto scene = m_sceneAssetManager->getScene("TestScene2");
    auto cameraGameObject = scene->getRootNode()->getChildren()[0]->getGameObject(0);
    auto camera = cameraGameObject->getComponent<Transform>();
    camera->setPosition({-4.0f, 4.0f, -18.0f});

    auto childs = scene->getRootNode()->getChildren();
    auto go = childs[0]->getGameObjects();
    for (auto &g : go) {
      if (g->getComponent<MeshFilter>() == nullptr)
        continue;
      auto t = g->getComponent<Transform>();
      auto r = t->getRotation();
      r.y += 0.01f;
      t->setRotation(r);
    }
    timing += 0.01f;

    auto sFrameStart = renderer->beginFrame();
    auto renderComplete =
        renderer->render(scene.get(), nullptr, renderTargets.get(), renderConfig, {sFrameStart.get()});
    renderer->endFrame({renderComplete.get()});
  }

  void onEnd() override {}
};
} // namespace Ifrit

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

  Ifrit::DemoApplicationAyanami app;
  app.run(info);
  return 0;
}