
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
#include "ifrit/core/math/LinalgOps.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/Runtime.h"
#include <numbers>
#include <thread>

#define WINDOW_WIDTH 1980
#define WINDOW_HEIGHT 1080

using namespace Ifrit;
using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::MeshProcLib::MeshProcess;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime;
using namespace Ifrit;

class CameraMovingScript : public ActorBehavior
{
    using ActorBehavior::ActorBehavior;

private:
    f32          m_movLeft   = 0.0f;
    f32          m_movRight  = 0.0f;
    f32          m_movTop    = 0.0f;
    f32          m_movBottom = 0.0f;
    f32          m_movFar    = 0.0f;
    f32          m_movNear   = 0.0f;
    f32          m_movRot    = 0.0f;

    InputSystem* m_inputSystem;

public:
    void SetInputSystem(InputSystem* inputSystem) { m_inputSystem = inputSystem; }

    void OnUpdate() override
    {
        auto scale       = 0.12f;
        auto inputSystem = m_inputSystem;
        if (inputSystem->IsKeyPressed(InputKeyCode::A))
            m_movLeft += scale;
        if (inputSystem->IsKeyPressed(InputKeyCode::D))
            m_movRight += scale;
        if (inputSystem->IsKeyPressed(InputKeyCode::W))
            m_movTop += scale;
        if (inputSystem->IsKeyPressed(InputKeyCode::S))
            m_movBottom += scale;
        if (inputSystem->IsKeyPressed(InputKeyCode::E))
            m_movFar += scale;
        if (inputSystem->IsKeyPressed(InputKeyCode::F))
            m_movNear += scale;
        if (inputSystem->IsKeyPressed(InputKeyCode::Z))
            m_movRot += scale * 0.03f;
        if (inputSystem->IsKeyPressed(InputKeyCode::X))
            m_movRot -= scale * 0.03f;

        auto parent = this->GetParentUnsafe();
        auto camera = parent->GetComponent<Transform>();
        if (camera)
        {
            camera->SetPosition(
                { -20.0f + m_movRight - m_movLeft, 8.0f + m_movTop - m_movBottom, 2.05f + m_movFar - m_movNear });
            camera->SetRotation({ 0.0f, m_movRot + 1.57f, 0.0f });
        }
    }
};

class DemoApplication : public Application
{
private:
    RhiScissor                     scissor = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
    Ref<RhiRenderTargets>          renderTargets;
    Ref<RhiColorAttachment>        colorAttachment;
    RhiTextureRef                  depthImage;
    Ref<RhiDepthStencilAttachment> depthAttachment;
    Ref<SyaroRenderer>             renderer;
    RhiTexture*                    swapchainImg;
    RendererConfig                 renderConfig;
    float                          timing = 0;

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

        auto cameraMover = cameraGameObject->AddComponent<CameraMovingScript>();
        cameraMover->SetInputSystem(m_inputSystem.get());

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

        auto meshes    = bistroObj->GetLoadedMesh(s.get());
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
        depthAttachment =
            rt->CreateRenderTargetDepthStencil(depthImage.get(), { {}, 1.0f }, RhiRenderTargetLoadOp::Clear);
        renderTargets->SetColorAttachments({ colorAttachment.get() });
        renderTargets->SetDepthStencilAttachment(depthAttachment.get());
        renderTargets->SetRenderArea(scissor);

        m_sceneManager->SetActiveScene(s);
    }

    void OnUpdate() override
    {
        auto scene       = m_sceneManager->GetActiveScene();
        timing           = timing + 0.1f;
        auto sFrameStart = renderer->BeginFrame();
        auto renderComplete =
            renderer->Render(scene.get(), nullptr, renderTargets.get(), renderConfig, { sFrameStart.get() });
        renderer->EndFrame({ renderComplete.get() });
    }

    void OnEnd() override {}
};

int main()
{
    ProjectProperty info;
    info.m_assetPath             = IFRIT_DEMO_ASSET_PATH;
    info.m_scenePath             = IFRIT_DEMO_SCENE_PATH;
    info.m_displayProvider       = AppDisplayProvider::GLFW;
    info.m_rhiType               = AppRhiType::Vulkan;
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