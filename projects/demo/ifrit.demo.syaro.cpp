
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

using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::MeshProcLib::MeshProcess;
using namespace Ifrit::Math;
using namespace Ifrit::Core;
using namespace Ifrit::Common::Utility;

// Glfw key function here
float movLeft = 0, movRight = 0, movTop = 0, movBottom = 0, movFar = 0, movNear = 0, movRot = 0;

class DemoApplication : public Ifrit::Core::Application
{
private:
    RhiScissor                            scissor = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
    Ifrit::Ref<RhiRenderTargets>          renderTargets;
    Ifrit::Ref<RhiColorAttachment>        colorAttachment;
    RhiTextureRef                         depthImage;
    Ifrit::Ref<RhiDepthStencilAttachment> depthAttachment;
    Ifrit::Ref<SyaroRenderer>             renderer;
    RhiTexture*                           swapchainImg;
    RendererConfig                        renderConfig;
    float                                 timing = 0;

public:
    void OnStart() override
    {
        renderer       = std::make_shared<SyaroRenderer>(this);
        auto bistroObj = m_assetManager->GetAssetByName<GLTFAsset>("Bistro/untitled.gltf");
        // Renderer config
        renderConfig.m_visualizationType          = RendererVisualizationType::Default;
        renderConfig.m_indirectLightingType       = IndirectLightingType::HBAO;
        renderConfig.m_antiAliasingType           = AntiAliasingType::TAA;
        renderConfig.m_shadowConfig.m_maxDistance = 200.0f;
        renderConfig.m_superSamplingRate          = 1.0f;

        // Scene
        auto s    = m_sceneAssetManager->CreateScene("TestScene2");
        auto node = s->AddSceneNode();

        auto cameraGameObject = node->AddGameObject("camera");
        auto camera           = cameraGameObject->AddComponent<Camera>();
        camera->SetCameraType(CameraType::Perspective);
        camera->SetMainCamera(true);
        camera->SetAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
        camera->SetFov(60.0f / 180.0f * std::numbers::pi_v<float>);
        camera->SetFar(200.0f);
        camera->SetNear(1.00f);

        auto cameraTransform = cameraGameObject->GetComponent<Transform>();
        cameraTransform->SetPosition({ 0.0f, 0.5f, -1.25f });
        cameraTransform->SetRotation({ 0.0f, 0.1f, 0.0f });
        cameraTransform->SetScale({ 1.0f, 1.0f, 1.0f });

        auto lightGameObject = node->AddGameObject("sun");
        auto light           = lightGameObject->AddComponent<Light>();
        auto lightTransform  = lightGameObject->GetComponent<Transform>();
        lightTransform->SetRotation({ 120.0 / 180.0f * std::numbers::pi_v<float>, 0.0f, 0.0f });
        light->SetShadowMap(true);
        light->SetShadowMapResolution(2048);
        light->SetAffectPbrSky(true);

        auto meshes    = bistroObj->GetLoadedMesh();
        auto numMeshes = 0;
        for (auto& m : meshes)
        {
            numMeshes++;
            node->AddGameObjectTransferred(std::move(m->m_prefab));
        }

        // Render targets
        auto rt       = m_rhiLayer.get();
        depthImage    = rt->CreateDepthTexture("Demo_Depth", WINDOW_WIDTH, WINDOW_HEIGHT, false);
        swapchainImg  = rt->GetSwapchainImage();
        renderTargets = rt->CreateRenderTargets();
        colorAttachment =
            rt->CreateRenderTarget(swapchainImg, { 0.0f, 0.0f, 0.0f, 1.0f }, RhiRenderTargetLoadOp::Clear, 0, 0);
        depthAttachment = rt->CreateRenderTarGetDepthStencil(depthImage.get(), { {}, 1.0f }, RhiRenderTargetLoadOp::Clear);
        renderTargets->SetColorAttachments({ colorAttachment.get() });
        renderTargets->SetDepthStencilAttachment(depthAttachment.get());
        renderTargets->SetRenderArea(scissor);

        m_sceneManager->SetActiveScene(s);
    }

    void OnUpdate() override
    {
        {
            auto scale       = 0.12f;
            auto inputSystem = m_inputSystem.get();
            if (inputSystem->IsKeyPressed(GLFW_KEY_A))
                movLeft += scale;
            if (inputSystem->IsKeyPressed(GLFW_KEY_D))
                movRight += scale;
            if (inputSystem->IsKeyPressed(GLFW_KEY_W))
                movTop += scale;
            if (inputSystem->IsKeyPressed(GLFW_KEY_S))
                movBottom += scale;
            if (inputSystem->IsKeyPressed(GLFW_KEY_E))
                movFar += scale;
            if (inputSystem->IsKeyPressed(GLFW_KEY_F))
                movNear += scale;
            if (inputSystem->IsKeyPressed(GLFW_KEY_Z))
                movRot += scale * 0.03f;
            if (inputSystem->IsKeyPressed(GLFW_KEY_X))
                movRot -= scale * 0.03f;
        }

        auto scene            = m_sceneManager->GetActiveScene();
        auto cameraGameObject = scene->GetRootNode()->GetChildren()[0]->GetGameObject(0);
        auto camera           = cameraGameObject->GetComponent<Transform>();
        timing                = timing + 0.1f;
        camera->SetPosition({ -20.0f + movRight - movLeft, 8.0f + movTop - movBottom, 2.05f + movFar - movNear });
        camera->SetRotation({ 0.0f, movRot + 1.57f, 0.0f });
        auto sFrameStart = renderer->BeginFrame();
        auto renderComplete =
            renderer->Render(scene.get(), nullptr, renderTargets.get(), renderConfig, { sFrameStart.get() });
        renderer->EndFrame({ renderComplete.get() });
    }

    void OnEnd() override {}
};

int main()
{
    Ifrit::Core::ProjectProperty info;
    info.m_assetPath             = IFRIT_DEMO_ASSET_PATH;
    info.m_scenePath             = IFRIT_DEMO_SCENE_PATH;
    info.m_displayProvider       = Ifrit::Core::AppDisplayProvider::GLFW;
    info.m_rhiType               = Ifrit::Core::AppRhiType::Vulkan;
    info.m_width                 = 1980;
    info.m_height                = 1080;
    info.m_rhiComputeQueueCount  = 1;
    info.m_rhiGraphicsQueueCount = 1;
    info.m_rhiTransferQueueCount = 1;
    info.m_rhiNumBackBuffers     = 2;
    info.m_name                  = "Ifrit-v2";
    info.m_cachePath             = IFRIT_DEMO_CACHE_PATH;
    info.m_rhiDebugMode          = true;

    DemoApplication app;
    app.Run(info);
    return 0;
}