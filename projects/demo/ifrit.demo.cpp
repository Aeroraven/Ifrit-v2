#ifndef IFRIT_DLL
#define IFRIT_DLL
#endif
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/Core.h"
#include "ifrit/display/presentation/window/GLFWWindowProvider.h"
#include <numbers>

#define WINDOW_WIDTH 1980
#define WINDOW_HEIGHT 1080

using namespace Ifrit::GraphicsBackend::Rhi;
using namespace Ifrit::MeshProcLib::ClusterLod;
using namespace Ifrit::Math;
using namespace Ifrit::Core;
using namespace Ifrit::Common::Utility;

// Glfw key function here
float movLeft = 0, movRight = 0, movTop = 0, movBottom = 0, movFar = 0,
      movNear = 0;

void key_callback(int key, int scancode, int action, int mods) {
  auto scale = 0.01f;
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
}

class DemoApplication : public Ifrit::Core::Application {
private:
  RhiScissor scissor = {0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
  std::shared_ptr<Material> m_material;
  std::shared_ptr<RhiRenderTargets> renderTargets;
  std::shared_ptr<RhiColorAttachment> colorAttachment;
  std::shared_ptr<RhiDepthStencilAttachment> depthAttachment;
  std::shared_ptr<SyaroRenderer> renderer;
  RhiTexture *swapchainImg;

  PerFrameData perframeData;
  float timing = 0;

  constexpr static std::array<ifloat3, 4> bunnyPositions = {
      ifloat3{-0.3f, 0.0f, 0.0f}, ifloat3{0.3f, 0.0f, 0.0f},
      ifloat3{0.0f, 0.0f, 0.0f}, ifloat3{0.0f, 0.0f, 25.5f}};

public:
  void onStart() override {
    renderer = std::make_shared<SyaroRenderer>(this);
    m_windowProvider->registerKeyCallback(key_callback);
    auto obj = m_assetManager->getAssetByName<WaveFrontAsset>("bunny.obj");
    auto meshShader = m_assetManager->getAssetByName<ShaderAsset>(
        "Shader/ifrit.mesh2.mesh.glsl");
    auto fragShader = m_assetManager->getAssetByName<ShaderAsset>(
        "Shader/ifrit.mesh2.frag.glsl");

    // Material
    m_material = std::make_shared<Material>();
    m_material->m_effectTemplates[GraphicsShaderPassType::Opaque] =
        ShaderEffect();
    auto &effect =
        m_material->m_effectTemplates[GraphicsShaderPassType::Opaque];
    effect.m_shaders.push_back(meshShader->loadShader());
    effect.m_shaders.push_back(fragShader->loadShader());

    // Scene
    auto s = m_sceneAssetManager->createScene("TestScene2");
    auto node = s->addSceneNode();

    for (int i = 0; i < bunnyPositions.size(); i++) {
      auto gameObject = node->addGameObject("bunny");
      auto meshFilter = gameObject->addComponent<MeshFilter>();
      meshFilter->setMesh(obj);
      auto meshRenderer = gameObject->addComponent<MeshRenderer>();
      meshRenderer->setMaterial(m_material);

      auto objectTransform = gameObject->addComponent<Transform>();
      objectTransform->setPosition(
          {bunnyPositions[i].x, bunnyPositions[i].y, bunnyPositions[i].z});
      objectTransform->setRotation({0.0f, 0.0f, 0.0f});
      objectTransform->setScale({1.0f, 1.0f, 1.0f});
    }

    auto cameraGameObject = node->addGameObject("camera");
    auto camera = cameraGameObject->addComponent<Camera>();
    camera->setMainCamera(true);
    camera->setAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
    camera->setFov(90.0f / 180.0f * std::numbers::pi_v<float>);
    camera->setFar(3000.0f);
    camera->setNear(0.01f);

    auto cameraTransform = cameraGameObject->getComponent<Transform>();
    cameraTransform->setPosition({0.0f, 0.1f, -1.25f});
    cameraTransform->setRotation({0.0f, 0.0f, 0.0f});
    cameraTransform->setScale({1.0f, 1.0f, 1.0f});

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
                                ->getGameObject(bunnyPositions.size());
    auto camera = cameraGameObject->getComponent<Transform>();
    timing = timing + 0.01f;
    camera->setPosition({0.0f + movRight - movLeft + 0.5f * std::sin(timing),
                         0.1f + movTop - movBottom,
                         -0.25f + movFar - movNear});
    auto sFrameStart = renderer->beginFrame();
    m_sceneManager->collectPerframeData(
        perframeData, m_sceneAssetManager->getScene("TestScene2").get());
    auto renderComplete = renderer->render(perframeData, renderTargets.get(),
                                           {sFrameStart.get()});
    renderer->endFrame({renderComplete.get()});
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

  DemoApplication app;
  app.run(info);
  return 0;
}