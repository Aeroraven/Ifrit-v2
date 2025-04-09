
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

#include "ifrit/core/logging/Logging.h"
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

// Glfw key function here
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
            m_movRot += scale * 0.2f;
        if (inputSystem->IsKeyPressed(InputKeyCode::X))
            m_movRot -= scale * 0.2f;

        auto parent = this->GetParentUnsafe();
        auto camera = parent->GetComponent<Transform>();
        if (camera)
        {
            // 11.519989 2.360000 -5.760006
            camera->SetPosition({ 3.239989f + m_movRight - m_movLeft, 2.240000f + m_movTop - m_movBottom,
                -6.000006f + m_movFar - m_movNear });
            camera->SetRotation({ 0.0f, m_movRot + 6.91f, 0.0f });

            // if print q, print the position and rotation
            if (m_inputSystem->IsKeyPressed(InputKeyCode::Q))
            {
                auto pos = camera->GetPosition();
                auto rot = camera->GetRotation();
                iInfo("Camera Position: {}, {}, {}", pos.x, pos.y, pos.z);
                iInfo("Camera Rotation: {}, {}, {}", rot.x, rot.y, rot.z);
            }
        }
    }
};

class LightRotScript : public ActorBehavior
{
    using ActorBehavior::ActorBehavior;

private:
    f32          m_rotX     = 13.0f;
    f32          m_rotY     = 2.0f;
    f32          m_rotZ     = 0.0f;
    f32          m_rotSpeed = 0.1f;

    InputSystem* m_inputSystem;

public:
    void SetInputSystem(InputSystem* inputSystem) { m_inputSystem = inputSystem; }
    void OnUpdate() override
    {
        auto parent = this->GetParentUnsafe();
        auto light  = parent->GetComponent<Transform>();
        if (light)
        {
            if (m_inputSystem->IsKeyPressed(InputKeyCode::Q))
            {
                auto pos = light->GetPosition();
                auto rot = light->GetRotation();
                iInfo("Light Position: {}, {}, {}", pos.x, pos.y, pos.z);
                iInfo("Light Rotation: {}, {}, {}", rot.x, rot.y, rot.z);
            }
            if (m_inputSystem->IsKeyPressed(InputKeyCode::U))
            {
                m_rotZ += m_rotSpeed;
            }
            if (m_inputSystem->IsKeyPressed(InputKeyCode::J))
            {
                m_rotX += m_rotSpeed;
            }
            if (m_inputSystem->IsKeyPressed(InputKeyCode::I))
            {
                m_rotY += m_rotSpeed;
            }
            light->SetRotation({ m_rotX, m_rotY, m_rotZ });
        }
    }
};

namespace Ifrit
{
    class DemoApplicationAyanami : public Runtime::Application
    {
    private:
        RhiScissor                     scissor = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
        Ref<RhiRenderTargets>          renderTargets;
        Ref<RhiColorAttachment>        colorAttachment;
        RhiTextureRef                  depthImage;
        Ref<RhiDepthStencilAttachment> depthAttachment;
        Ref<AyanamiRenderer>           renderer;
        RhiTexture*                    swapchainImg;
        RendererConfig                 renderConfig;
        float                          timing = 0;

    public:
        void OnStart() override
        {
            iInfo("DemoApplication::OnStart()");

            Ayanami::AyanamiRenderConfig ayaConfig;
            ayaConfig.m_globalDFClipmapLevels       = 1;
            ayaConfig.m_globalDFClipmapResolution   = 256;
            ayaConfig.m_globalDFBaseExtent          = 14.0f;
            ayaConfig.m_DebugForceSurfaceCacheRegen = false;

            renderer       = std::make_shared<AyanamiRenderer>(this, ayaConfig);
            auto bistroObj = m_assetManager->GetAssetByName<GLTFAsset>("BistroInterior/Untitled.gltf"); //
            // auto bistroObj = m_assetManager->GetAssetByName<GLTFAsset>("Fox/scene.gltf"); //
            //   Scene
            auto s    = m_sceneAssetManager->CreateScene("TestScene2");
            auto node = s->AddSceneNode();

            auto cameraGameObject = node->AddGameObject("camera");
            auto camera           = cameraGameObject->AddComponent<Camera>();
            camera->SetCameraType(CameraType::Perspective);
            camera->SetMainCamera(true);
            camera->SetAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
            camera->SetFov(60.0f / 180.0f * std::numbers::pi_v<float>);
            camera->SetFar(60.0f);
            camera->SetNear(0.50f);

            auto cameraTransform = cameraGameObject->GetComponent<Transform>();
            cameraTransform->SetPosition({ 0.0f, 0.5f, -1.25f });
            cameraTransform->SetRotation({ 0.0f, 0.1f, 0.0f });
            cameraTransform->SetScale({ 1.0f, 1.0f, 1.0f });
            cameraTransform->SetPosition({ 0.0f, 2.0f, -0.0f });

            auto cameraMover = cameraGameObject->AddComponent<CameraMovingScript>();
            cameraMover->SetInputSystem(m_inputSystem.get());

            auto lightGameObject = node->AddGameObject("sun");
            auto light           = lightGameObject->AddComponent<Light>();
            auto lightTransform  = lightGameObject->GetComponent<Transform>();
            lightTransform->SetRotation({ 60.0 / 180.0f * std::numbers::pi_v<float>, 0.0f, 0.0f });
            light->SetShadowMap(true);
            light->SetShadowMapResolution(2048);
            light->SetAffectPbrSky(true);

            auto lightRotScript = lightGameObject->AddComponent<LightRotScript>();
            lightRotScript->SetInputSystem(m_inputSystem.get());

            auto meshes = bistroObj->GetLoadedMesh(s.get());

            auto numMeshes = 0;
            for (auto& m : meshes)
            {
                numMeshes++;
                if (numMeshes <= 631)
                    continue;
                if (numMeshes >= 700 && numMeshes < 750)
                    continue;
                if (numMeshes > 812)
                    break;
                auto t      = m->m_prefab;
                auto meshDF = t->AddComponent<Ayanami::AyanamiMeshDF>();
                meshDF->BuildMeshDF(GetCacheDir());
                auto meshMarker = t->AddComponent<Ayanami::AyanamiMeshMarker>();

                auto transform = t->GetComponent<Transform>();
                transform->SetRotation({ 0.0f, 0.0f, 0.0f });
                transform->SetPosition({ 0.0f, 0.0f, 0.0f });
                transform->SetScale({ 0.01f, 0.01f, 0.01f });
                auto mat = transform->GetModelToWorldMatrix();
                node->AddGameObjectTransferred(std::move(m->m_prefab));
            }
            iInfo("Num meshes: {}", numMeshes);
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
            auto sFrameStart = renderer->BeginFrame();
            auto renderComplete =
                renderer->Render(scene.get(), nullptr, renderTargets.get(), renderConfig, { sFrameStart.get() });
            renderer->EndFrame({ renderComplete.get() });
            // std::abort();
        }

        void OnEnd() override {}
    };
} // namespace Ifrit

int main()
{
    using namespace Ifrit;

    Runtime::ProjectProperty info;
    info.m_assetPath             = IFRIT_DEMO_ASSET_PATH;
    info.m_scenePath             = IFRIT_DEMO_SCENE_PATH;
    info.m_displayProvider       = Runtime::AppDisplayProvider::GLFW;
    info.m_rhiType               = Runtime::AppRhiType::Vulkan;
    info.m_width                 = 1980;
    info.m_height                = 1080;
    info.m_rhiComputeQueueCount  = 1;
    info.m_rhiGraphicsQueueCount = 1;
    info.m_rhiTransferQueueCount = 1;
    info.m_rhiNumBackBuffers     = 2;
    info.m_name                  = "Ifrit-v2";
    info.m_cachePath             = IFRIT_DEMO_CACHE_PATH;
    info.m_rhiDebugMode          = true;

    DemoApplicationAyanami app;
    app.Run(info);
    return 0;
}