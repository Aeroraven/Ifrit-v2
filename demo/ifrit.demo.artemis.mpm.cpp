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
#include "ifrit/editor/EditorProviderHelper.h"
#include "ifrit/core/hal/HalDisplay.h"

#include "ifrit/runtime/geometry/preset/Circle2D.h"
#include "ifrit/runtime/geometry/preset/Square2D.h"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 800

using namespace Ifrit;
using namespace Ifrit::RHI;
using namespace GeometryProc;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime;
using namespace Ifrit::GeometryProc;

namespace Ifrit
{
    inline constexpr f32 kDefaultTimestep = 100.0f;
    static f32           sTimestep        = 1.0f / kDefaultTimestep;

    class MPMTiming : public ActorBehavior
    {
        using ActorBehavior::ActorBehavior;

    private:
        typedef ActorBehavior Super;
        f32                   m_InvTimestep = kDefaultTimestep;

    public:
        void SetupProperties() override
        {
            AddProperty<f32, EPropertyEditorType::Range>("Time Interval", m_InvTimestep, 60.0f, 2000.0f, 0.001f);
        }
        void OnUpdate() override { sTimestep = 1.0f / m_InvTimestep; }
    };

    class DemoApplicationMpm : public Runtime::Application
    {
    private:
        Owner<BaseForwardRenderer>                   m_Renderer;
        RendererConfig                               m_RenderConfig;

        Owner<Artemis::MPMSimulator>                 m_MpmSim;
        Owner<Artemis::ArtemisSimulator>             m_ArtemisSim;
        Owner<Artemis::RigidSimulator>               m_RigidSim;
        Vec<Vector3f>                                m_PointClouds;

        u32                                          m_FrameIdx = 0;
        Ref<FrameGraphResourcePool>                  m_FrameGraphResourcePool;
        // Debug
        Ref<Geometry::ParticleSurfaceProceduralMesh> m_ParticleSurfaceMesh;

    public:
        void OnStart() override
        {
            m_Renderer                               = MakeOwner<BaseForwardRenderer>(this);
            m_RenderConfig.m_AntiAliasingType        = AntiAliasingType::None;
            m_RenderConfig.m_OverrideMaterialCulling = OverrideMaterialCulling::ForcedCullNone;

            m_MpmSim     = MakeOwner<Artemis::MPMSimulator>();
            m_RigidSim   = MakeOwner<Artemis::RigidSimulator>();
            m_ArtemisSim = MakeOwner<Artemis::ArtemisSimulator>(this);

            m_ArtemisSim->RegisterSolver(m_MpmSim.get());
            // m_ArtemisSim->RegisterSolver(m_RigidSim.get());

            RegisterSubsystem(Editor::CreateEditorProvider(Editor::EEditorProviderType::ImGui));
            EnableRendererWrapper(true);

            {
                auto vdbFileData = Ifrit::ReadBinaryFile(IFRIT_DEMO_ASSET_PATH "/bunny.vdb");
                auto vdbDesc     = VDB::LoadVdbFromString(vdbFileData);
                VDB::PrintVdbMeta(vdbDesc);
                m_PointClouds = VDB::PoissonSampleVdbZpcReference(vdbDesc, 0.3f, 10);
                // iDebug("Sampled {} points from VDB.", m_PointClouds.size());
                PointCloud::PointCloudDescriptor pcDesc;
                pcDesc.m_Points = m_PointClouds.data();
                pcDesc.m_Count  = static_cast<u32>(m_PointClouds.size());

                PointCloud::MoveCenterTo(pcDesc, Vector3f(32.0f, 32.0f, 32.0f));
                PointCloud::NormalizeToLongestAxisAABB(pcDesc, Vector3f(0.0f), Vector3f(64.0f));
                // m_MpmSim->SetInitParticleLocations<3>(m_PointClouds);
            }

            auto scene               = m_sceneAssetManager->CreateScene("TestScene2");
            auto node                = scene->AddSceneNode();
            m_FrameGraphResourcePool = MakeRef<FrameGraphResourcePool>(GetRhi());

            auto timeControl = node->AddGameObject("MPMTimeControl");
            timeControl->AddComponent<MPMTiming>();

            auto mpmGlobalConfig = node->AddGameObject("MPMGlobalConfig");
            auto mpmConfig       = mpmGlobalConfig->AddComponent<Artemis::MPMSimulatorConfigurator>();
            mpmConfig->SetActiveSimulator(m_MpmSim.get());

            auto mpmContainer          = node->AddGameObject("MPMParticleContainer");
            auto mpmContainerComponent = mpmContainer->AddComponent<Artemis::MPMParticleContainer>();

            auto cameraGameObject = node->AddGameObject("Camera");
            auto camera           = cameraGameObject->AddComponent<Camera>();
            camera->SetCameraType(CameraType::Orthographic);
            camera->SetMainCamera(true);
            camera->SetAspect(1.0f * WINDOW_WIDTH / WINDOW_HEIGHT);
            camera->SetOrthoSpaceSize(1.0f);
            camera->SetFar(20.0f);
            camera->SetNear(0.10f);

            auto cameraTransform = cameraGameObject->GetComponent<Transform>();
            cameraTransform->SetScale({ 1.0f, 1.0f, 1.0f });
            cameraTransform->SetPosition({ 0.5f, 0.5f, -1.0f });

            auto material = MakeRef<DefaultMaterial>(this);
            material->BuildMaterial();

            auto rigid      = node->AddGameObject("RigidCollider");
            auto circleMesh = MakeRef<Geometry::Square2D>(0.1f, 0.1f);
            auto rigidMesh  = rigid->AddComponent<MeshFilter>();
            rigidMesh->SetMesh(circleMesh);
            auto rigidRenderer = rigid->AddComponent<MeshRenderer>();
            rigidRenderer->SetMaterial(material);
            auto rigidTransform = rigid->GetComponent<Transform>();
            rigidTransform->SetPosition({ 0.5f, 0.8f, 0.0f });
            rigidTransform->SetDevice(TransformUpdateDevice::GPU);
            auto rigidCollider = rigid->AddComponent<Artemis::GPURigidCollider>();
            rigidCollider->SetRadius(0.05f);
            rigidCollider->SetColliderType(Artemis::GPURigidColliderType::Box);
            rigidCollider->SetEnable(false);

            auto defaultEmitter = node->AddGameObject("ParticleEmitter");
            auto emitter        = defaultEmitter->AddComponent<Artemis::MPMParticleEmitter>();
            if (GetApplicationState()->m_EditorMode)
            {
                emitter->SetEnable(false);
            }

            m_sceneManager->SetActiveScene(scene);
            m_RendererWrapper->SetRenderer(m_Renderer.get());

            auto sceneSerialized = scene->Serialize();
            WriteTextFile("C:/WR/Test.json", sceneSerialized);
        }

        void OnUpdate() override
        {
            m_FrameIdx++;

            m_RendererWrapper->EnqueueRendererTask(m_sceneManager->GetActiveScene().get(), nullptr,
                m_RendererWrapper->GetDefaultRenderTargets(), m_RenderConfig);
            m_MpmSim->SetDebugRenderTarget(m_RendererWrapper->GetDefaultColorImage().get());
            m_ArtemisSim->CollectScene(m_sceneManager->GetActiveScene().get());
            m_RendererWrapper->EnqueueGeneralTask(
                [&](RHI::RhiTaskSubmission* submission) { return m_ArtemisSim->Update(sTimestep, { submission }); });
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

    info.m_EnableDPIScaling = true;

    DemoApplicationMpm app;
    app.Run(info);
    return 0;
}
