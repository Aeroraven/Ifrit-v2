#ifndef IFRIT_DLL
    #define IFRIT_DLL
#endif

#ifndef IFRIT_DEMO_ASSET_PATH
    #define IFRIT_DEMO_ASSET_PATH ""
#endif

#ifndef IFRIT_DEMO_CACHE_PATH
    #define IFRIT_DEMO_CACHE_PATH ""
#endif

#ifndef IFRIT_DEMO_SCENE_PATH
    #define IFRIT_DEMO_SCENE_PATH ""
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
#include "ifrit/runtime/physics/artemis/mpm/MPMSimulator.h"

#include "ifrit/ui/UIProviderHelper.h"

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
    static f32 sTimestep = 1.0f / 1500.0f;

    class MpmConfigurator : public ActorBehavior
    {
        using ActorBehavior::ActorBehavior;

    private:
        typedef ActorBehavior Super;
        f32                   m_InvTimestep = 1500.0f;

    public:
        void SetupProperties() override
        {
            AddProperty<f32, EPropertyEditorType::Range>("Time Interval", m_InvTimestep, 500.0f, 5000.0f, 0.001f);
        }

        void OnUpdate() override { sTimestep = 1.0f / m_InvTimestep; }
    };

    class DemoApplicationMpm : public Runtime::Application
    {
    private:
        RhiScissor                                   scissor = { 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT };
        Ref<RhiRenderTargets>                        renderTargets;
        Ref<RhiColorAttachment>                      colorAttachment;
        RhiTextureRef                                depthImage;
        Ref<RhiDepthStencilAttachment>               depthAttachment;
        Ref<BaseForwardRenderer>                     renderer;
        RhiTexture*                                  swapchainImg;
        RendererConfig                               renderConfig;

        Ref<Artemis::MPMSimulator>                   m_MpmSim;

        Ref<FrameGraphCompiler>                      m_FrameGraphCompiler;
        Ref<FrameGraphExecutor>                      m_FrameGraphExecutor;
        Ref<FrameGraphResourcePool>                  m_FrameGraphResourcePool;
        Vec<Vector3f>                                m_PointClouds;

        u32                                          m_FrameIdx = 0;

        // Debug
        Ref<Geometry::ParticleSurfaceProceduralMesh> m_ParticleSurfaceMesh;

    public:
        void OnStart() override
        {
            iInfo("DemoApplication::OnStart()");
            renderer = MakeRef<BaseForwardRenderer>(this);
            m_MpmSim = MakeRef<Artemis::MPMSimulator>();

            RegisterSubsystem(UI::CreateUIProvider(UI::EUIProviderType::ImGui));
            EnableRendererWrapper(true);

            {
                auto vdbFileData = Ifrit::ReadBinaryFile(IFRIT_DEMO_ASSET_PATH "/bunny.vdb");
                auto vdbDesc     = VDB::LoadVdbFromString(vdbFileData);
                VDB::PrintVdbMeta(vdbDesc);
                m_PointClouds = VDB::PoissonSampleVdbZpcReference(vdbDesc, 0.3f, 10);
                iDebug("Sampled {} points from VDB.", m_PointClouds.size());
                PointCloud::PointCloudDescriptor pcDesc;
                pcDesc.m_Points = m_PointClouds.data();
                pcDesc.m_Count  = static_cast<u32>(m_PointClouds.size());

                PointCloud::MoveCenterTo(pcDesc, Vector3f(32.0f, 32.0f, 32.0f));
                PointCloud::NormalizeToLongestAxisAABB(pcDesc, Vector3f(0.0f), Vector3f(64.0f));
                // m_MpmSim->SetInitParticleLocations<3>(m_PointClouds);
            }

            renderConfig.m_ShadowConfig.m_maxDistance = 20.0f;
            renderConfig.m_AntiAliasingType           = AntiAliasingType::None;
            renderConfig.m_OverrideMaterialCulling    = OverrideMaterialCulling::ForcedCullNone;

            auto scene               = m_sceneAssetManager->CreateScene("TestScene2");
            auto node                = scene->AddSceneNode();
            m_FrameGraphCompiler     = MakeRef<FrameGraphCompiler>();
            m_FrameGraphExecutor     = MakeRef<FrameGraphExecutor>(GetRhi());
            m_FrameGraphResourcePool = MakeRef<FrameGraphResourcePool>(GetRhi());

            auto configurator = node->AddGameObject("Configurator");
            configurator->AddComponent<MpmConfigurator>();

            auto cameraGameObject = node->AddGameObject("Camera");
            auto camera           = cameraGameObject->AddComponent<Camera>();
            camera->SetCameraType(CameraType::Perspective);
            camera->SetMainCamera(true);
            camera->SetAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
            camera->SetFov(60.0f / 180.0f * std::numbers::pi_v<float>);
            camera->SetFar(20.0f);
            camera->SetNear(0.10f);

            auto cameraTransform = cameraGameObject->GetComponent<Transform>();
            cameraTransform->SetScale({ 1.0f, 1.0f, 1.0f });
            cameraTransform->SetPosition({ 0.5f, 0.5f, -1.0f });

            auto material = MakeRef<SyaroDefaultGBufEmitter>(this);
            material->BuildMaterial();
            auto meshingObject    = node->AddGameObject("Meshing");
            m_ParticleSurfaceMesh = MakeRef<Geometry::ParticleSurfaceProceduralMesh>();
            m_ParticleSurfaceMesh->Init(GetRhi(), 2145141, 2145141, Vector4i(200, 200, 200, 0),
                Vector3f(-0.01f, -0.01f, -0.01f), Vector3f(1.01f, 1.01f, 1.01f));
            auto meshFilter = meshingObject->AddComponent<MeshFilter>();
            meshFilter->SetMesh(m_ParticleSurfaceMesh);
            auto meshRenderer = meshingObject->AddComponent<MeshRenderer>();
            meshRenderer->SetMaterial(material);

            // Render targets
            auto rt         = GetRhi();
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

            m_RendererWrapper->SetRenderer(renderer.get());
        }

        void OnUpdate() override
        {
            m_FrameIdx++;
            if (m_FrameIdx == 511)
            {
                Artemis::MPMParticleEmitArgs args;
                args.m_MaterialType = Artemis::MPMSimulatorParticleType::Jelly;
                // m_MpmSim->EmitParticles<3>(m_PointClouds, args);
                // m_MpmSim->SetInitParticleLocations<3>(m_PointClouds);
            }

            m_ParticleSurfaceMesh->SetParticleData(
                m_MpmSim->GetParticlePositionBuffer(), m_MpmSim->GetParticleCounterBuffer());

            m_RendererWrapper->EnqueueRDGTask(
                [&](FrameGraphBuilder* builder) {
                    auto rt = &builder->ImportTexture("Demo_Swapchain", swapchainImg);
                    m_MpmSim->RunSolverStep(*builder, sTimestep);
                    // iDebug("Timestep: {}", sTimestep);
                    m_MpmSim->Render(*builder, rt);
                },
                m_FrameGraphResourcePool.get());

            if (0)
            {
                m_RendererWrapper->EnqueueRendererTask(
                    m_sceneManager->GetActiveScene().get(), nullptr, renderTargets.get(), renderConfig);
            }
        }

        void OnEnd() override {}
    };
} // namespace Ifrit

int main()
{
    using namespace Ifrit;

    Runtime::ProjectProperty info;
    info.m_assetPath       = IFRIT_DEMO_ASSET_PATH;
    info.m_scenePath       = IFRIT_DEMO_SCENE_PATH;
    info.m_displayProvider = Runtime::AppDisplayProvider::GLFW;
    info.m_rhiType         = Runtime::AppRhiType::Vulkan;
    info.m_width           = WINDOW_WIDTH;
    info.m_height          = WINDOW_HEIGHT;
    info.m_name            = "Ifrit-v2";
    info.m_cachePath       = IFRIT_DEMO_CACHE_PATH;
    info.m_rhiDebugMode    = true;

    DemoApplicationMpm app;
    app.Run(info);
    return 0;
}
