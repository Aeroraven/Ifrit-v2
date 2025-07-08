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
#include "ifrit/geomproc/vdb/VdbSampler.h"
#include "ifrit/geomproc/pointcloud/PointCloudTransforms.h"
#include "ifrit/runtime/physics/siro/mpm/MPMSimulator.h"

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600

using namespace Ifrit;
using namespace Ifrit::RHI;
using namespace GeometryProc;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime;
using namespace Ifrit::GeometryProc;

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

        Ref<Siro::MPMSimulator>        m_MpmSim;

        Ref<FrameGraphCompiler>        m_FrameGraphCompiler;
        Ref<FrameGraphExecutor>        m_FrameGraphExecutor;
        Ref<FrameGraphResourcePool>    m_FrameGraphResourcePool;

    public:
        void OnStart() override
        {
            iInfo("DemoApplication::OnStart()");
            renderer = MakeRef<BaseForwardRenderer>(this);
            m_MpmSim = MakeRef<Siro::MPMSimulator>();

            {
                auto vdbFileData = Ifrit::ReadBinaryFile(IFRIT_DEMO_ASSET_PATH "/bunny.vdb");
                auto vdbDesc     = VDB::LoadVdbFromString(vdbFileData);
                VDB::PrintVdbMeta(vdbDesc);
                auto                             p = VDB::PoissonSampleVdbZpcReference(vdbDesc, 0.5f, 8);
                // std::cout << "Sampled " << p.size() << " points from VDB." << std::endl;
                PointCloud::PointCloudDescriptor pcDesc;
                pcDesc.m_Points = p.data();
                pcDesc.m_Count  = static_cast<u32>(p.size());

                PointCloud::MoveCenterTo(pcDesc, Vector3f(32.0f, 32.0f, 32.0f));
                PointCloud::NormalizeToLongestAxisAABB(pcDesc, Vector3f(0.0f), Vector3f(64.0f));
                // m_MpmSim->SetInitParticleLocations<3>(p);
            }

            renderConfig.m_ShadowConfig.m_maxDistance = 20.0f;
            renderConfig.m_AntiAliasingType           = AntiAliasingType::None;
            renderConfig.m_OverrideMaterialCulling    = OverrideMaterialCulling::ForcedCullNone;

            auto scene = m_sceneAssetManager->CreateScene("TestScene2");
            auto node  = scene->AddSceneNode();

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
                    m_MpmSim->RunSolverStep(builder, 1.0f / 1500.0f);
                    m_MpmSim->Render(builder, &rt);

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
