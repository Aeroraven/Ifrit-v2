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
#include "ifrit/runtime/physics/artemis/mpm/MPMParticleEmitter.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMSimulatorConfigurator.h"
#include "ifrit/editor/EditorProviderHelper.h"

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 768

using namespace Ifrit;
using namespace Ifrit::RHI;
using namespace GeometryProc;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime;
using namespace Ifrit::GeometryProc;

namespace Ifrit
{
    static f32 sTimestep = 1.0f / 1500.0f;

    class MpmTiming : public ActorBehavior
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
        Ref<BaseForwardRenderer>                     renderer;
        RendererConfig                               renderConfig;

        Ref<Artemis::MPMSimulator>                   m_MpmSim;
        Ref<FrameGraphResourcePool>                  m_FrameGraphResourcePool;
        Vec<Vector3f>                                m_PointClouds;

        u32                                          m_FrameIdx = 0;

        // Debug
        Ref<Geometry::ParticleSurfaceProceduralMesh> m_ParticleSurfaceMesh;

    public:
        void OnStart() override
        {
            renderer = MakeRef<BaseForwardRenderer>(this);
            m_MpmSim = MakeRef<Artemis::MPMSimulator>();

            RegisterSubsystem(Editor::CreateEditorProvider(Editor::EEditorProviderType::ImGui));
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
            m_FrameGraphResourcePool = MakeRef<FrameGraphResourcePool>(GetRhi());

            auto timeControl = node->AddGameObject("MPMTimeControl");
            timeControl->AddComponent<MpmTiming>();

            auto mpmGlobalConfig = node->AddGameObject("MPMGlobalConfig");
            auto mpmConfig       = mpmGlobalConfig->AddComponent<Artemis::MPMSimulatorConfigurator>();
            mpmConfig->SetActiveSimulator(m_MpmSim.get());

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
            auto meshingObject    = node->AddGameObject("ProceduralMesh");
            m_ParticleSurfaceMesh = MakeRef<Geometry::ParticleSurfaceProceduralMesh>();
            m_ParticleSurfaceMesh->Init(GetRhi(), 2145141, 2145141, Vector4i(200, 200, 200, 0),
                Vector3f(-0.01f, -0.01f, -0.01f), Vector3f(1.01f, 1.01f, 1.01f));
            auto meshFilter = meshingObject->AddComponent<MeshFilter>();
            meshFilter->SetMesh(m_ParticleSurfaceMesh);
            auto meshRenderer = meshingObject->AddComponent<MeshRenderer>();
            meshRenderer->SetMaterial(material);

            auto defaultEmitter = node->AddGameObject("ParticleEmitter");
            auto emitter        = defaultEmitter->AddComponent<Artemis::MPMParticleEmitter>();
            if (GetApplicationState()->m_EditorMode)
            {
                emitter->SetEnable(false);
            }

            m_sceneManager->SetActiveScene(scene);
            m_RendererWrapper->SetRenderer(renderer.get());
        }

        void OnUpdate() override
        {
            m_FrameIdx++;
            m_ParticleSurfaceMesh->SetParticleData(
                m_MpmSim->GetParticlePositionBuffer(), m_MpmSim->GetParticleCounterBuffer());

            m_RendererWrapper->EnqueueRDGTask(
                [&](FrameGraphBuilder* builder) {
                    auto rt = &builder->ImportTexture("DemoRT", m_RendererWrapper->GetDefaultColorImage().get());
                    m_MpmSim->CollectScene(m_sceneManager->GetActiveScene().get());
                    m_MpmSim->RunSolverStep(*builder, sTimestep);
                    m_MpmSim->Render(*builder, rt);
                },
                m_FrameGraphResourcePool.get());

            if (0)
            {
                m_RendererWrapper->EnqueueRendererTask(m_sceneManager->GetActiveScene().get(), nullptr,
                    m_RendererWrapper->GetDefaultRenderTargets(), renderConfig);
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
