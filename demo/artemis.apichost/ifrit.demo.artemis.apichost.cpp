
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
#include "ifrit/core/math/linalg/LinalgOps.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/Runtime.h"
#include "ifrit/runtime/material/SyaroDefaultGBufEmitter.h"
#include <numbers>
#include <thread>
#include "ifrit/rhi/common/RhiStructHelper.h"

#include "ifrit/runtime/physics/artemis/apic/APICFluid.h"

#define WINDOW_WIDTH 1980
#define WINDOW_HEIGHT 1080

using namespace Ifrit;
using namespace Ifrit::RHI;
using namespace Ifrit::GeometryProc::MeshProcess;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime;

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
        Ref<BaseForwardRenderer>       renderer;
        RhiTexture*                    swapchainImg;
        RendererConfig                 renderConfig;

        Ref<Artemis::APICFluid>        m_ApicHostTest;

        Ref<FrameGraphCompiler>        m_FrameGraphCompiler;
        Ref<FrameGraphExecutor>        m_FrameGraphExecutor;
        Ref<FrameGraphResourcePool>    m_FrameGraphResourcePool;

    public:
        void OnStart() override
        {

            renderConfig.m_ShadowConfig.m_maxDistance = 20.0f;
            renderConfig.m_AntiAliasingType           = AntiAliasingType::None;
            renderConfig.m_OverrideMaterialCulling    = OverrideMaterialCulling::ForcedCullNone;

            renderer       = MakeRef<BaseForwardRenderer>(this);
            m_ApicHostTest = MakeRef<Artemis::APICFluid>();
            auto scene     = m_sceneAssetManager->CreateScene("TestScene2");
            auto node      = scene->AddSceneNode();

            m_FrameGraphCompiler     = MakeRef<FrameGraphCompiler>();
            m_FrameGraphExecutor     = MakeRef<FrameGraphExecutor>(GetRhi());
            m_FrameGraphResourcePool = MakeRef<FrameGraphResourcePool>(GetRhi());

            auto cameraGameObject = node->AddGameObject("camera");
            auto camera           = cameraGameObject->AddComponent<Camera>();
            camera->SetCameraType(CameraType::Perspective);
            camera->SetMainCamera(true);
            camera->SetAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
            camera->SetFov(60.0f / 180.0f * std::numbers::pi_v<float>);
            camera->SetFar(20.0f);
            camera->SetNear(0.10f);

            // Render targets
            auto rt         = m_rhiLayer.get();
            depthImage      = rt->CreateDepthTexture("Demo_Depth", WINDOW_WIDTH, WINDOW_HEIGHT, false);
            swapchainImg    = rt->GetSwapchainImage();
            renderTargets   = rt->CreateRenderTargets();
            colorAttachment = rt->CreateRenderTarget(
                swapchainImg, RHI::CreateRhiClearColorValue(Vector4f(0.0f)), RhiRenderTargetLoadOp::Clear, 0, 0);
            depthAttachment = rt->CreateRenderTargetDepthStencil(
                depthImage.get(), RHI::CreateRhiClearDepthStencilValue(1.0f, 0), RhiRenderTargetLoadOp::Clear);
            renderTargets->SetColorAttachments({ colorAttachment.get() });
            renderTargets->SetDepthStencilAttachment(depthAttachment.get());
            renderTargets->SetRenderArea(scissor);

            m_sceneManager->SetActiveScene(scene);
        }

        void OnUpdate() override
        {
            auto scene       = m_sceneManager->GetActiveScene();
            auto sFrameStart = renderer->BeginFrame();

            auto rhi  = GetRhi();
            auto dq   = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Graphics);
            auto task = dq->RunAsyncCommand(
                [&](const RhiCommandList* cmd) {
                    FrameGraphBuilder builder(GetShaderRegistry(), GetRhi(), m_FrameGraphResourcePool.get());
                    auto              rt = builder.ImportTexture("Demo_Swapchain", swapchainImg);
                    m_ApicHostTest->RunSolver(builder, &rt);

                    auto fg = m_FrameGraphCompiler->Compile(builder);
                    m_FrameGraphExecutor->ExecuteInSingleCmd(cmd, fg);
                },
                { sFrameStart.get() }, {});

            renderer->EndFrame({ task.get() });
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
    info.m_width                 = WINDOW_WIDTH;
    info.m_height                = WINDOW_HEIGHT;
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
