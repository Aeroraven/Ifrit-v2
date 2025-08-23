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
#include "ifrit/runtime/geometry/preset/Plane.h"
#include "ifrit/runtime/base/MeshComponent.h"
#include "MPMTiming.h"
#include "MPMMouseInteractor.h"
#include "RigidEmitter.h"
#include "MPMMeshingCfg.h"
#include "artemis.mpm.generated.h"

#include "ifrit/core/reflection/SerializeHelper.h"
#include "ifrit/runtime/physics/artemis/ArtemisController.h"
#include "ifrit/runtime/physics/artemis/mpm/visualization/ProceduralMPMMeshAsset.h"
#include "ifrit/runtime/geometry/ProceduralMeshUpdater.h"
#include "ifrit/profiler/ProfilerSystem.h"

#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"

#define WINDOW_WIDTH 1500
#define WINDOW_HEIGHT 800

using namespace Ifrit;
using namespace Ifrit::RHI;
using namespace GeometryProc;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime;
using namespace Ifrit::GeometryProc;

namespace Ifrit
{

    class DemoApplicationMpm : public Runtime::Application
    {
    private:
        Owner<RendererBase>        m_Renderer;
        RendererConfig             m_RenderConfig;

        Artemis::GPURigidCollider* collider1;
        Artemis::GPURigidCollider* collider2;
        SceneNode*                 nodew;

    public:
        void OnStart() override
        {
            bool use3D   = true;
            auto meshRes = 96;
            if (use3D)
            {
                m_Renderer = MakeOwner<BaseDeferredRenderer>(this);
            }
            else
            {
                m_Renderer = MakeOwner<BaseForwardRenderer>(this);
            }

            m_RendererWrapper->SetRenderer(m_Renderer.get());

            RegisterSubsystem(InputSystem::Create());
            RegisterSubsystem(Artemis::ArtemisController::Create());
            RegisterSubsystem(Profiler::ProfilerSystem::Create());
            RegisterSubsystem(Geometry::ProceduralMeshUpdater::Create());
            RegisterSubsystem(Editor::CreateEditorProvider(Editor::EEditorProviderType::ImGui));
            EnableRendererWrapper(true);
            m_RendererWrapper->SetRendererConfig(m_RenderConfig);

            auto artemisController = GetSubsystem<Artemis::ArtemisController>();
            artemisController->AddPresetSolver(Artemis::EPresetArtemisSimulator::MPM);
            auto mpmSimulator = reinterpret_cast<Artemis::MPMSimulator*>(
                artemisController->GetPresetSolver(Artemis::EPresetArtemisSimulator::MPM));
            auto mpmInternalConfig = mpmSimulator->GetActiveConfig();
            if (use3D)
            {
                mpmInternalConfig.m_Dimension           = Artemis::MPMSimulatorProblemDimension::ThreeDimensional;
                mpmInternalConfig.m_DefaultParticleType = Artemis::MPMSimulatorParticleType::Jelly;
                mpmInternalConfig.m_GridSize            = Vector3u(64, 64, 64);
                mpmInternalConfig.m_GridSpacing         = 1.0f / 64.0f;
                mpmInternalConfig.m_DefaultMass         = 0.5f / 64.0f;
            }
            else
            {
                mpmInternalConfig.m_Dimension   = Artemis::MPMSimulatorProblemDimension::TwoDimensional;
                mpmInternalConfig.m_GridSize    = Vector3u(128, 128, 128);
                mpmInternalConfig.m_GridSpacing = 1.0f / 128;
                mpmInternalConfig.m_DefaultMass = 0.5f / 128;
            }

            mpmSimulator->SetConfig(mpmInternalConfig);

            // Asset
            auto mpmMesh = m_assetManager->CreateAsset<Artemis::ProceduralMPMMeshAsset>(String("SurfaceMesh"), 2145141u,
                2145141u, Vector4i(meshRes, meshRes, meshRes, 0), Vector3f(-0.01f, -0.01f, -0.01f),
                Vector3f(1.01f, 1.01f, 1.01f));
            auto material = m_assetManager->CreateAsset<DefaultMaterialAsset>(String("SurfaceMat"));
            auto plane    = m_assetManager->CreateAsset<Geometry::PlaneAsset>(String("Plane"), 1.0f, 1.0f);

            // Scene
            auto scene = m_sceneAssetManager->CreateScene("TestScene2");
            auto node  = scene->AddSceneNode("MPMScene");
            nodew      = node;

            auto mpmGlobalConfig       = node->AddGameObject("MPMControl");
            auto mpmConfig             = mpmGlobalConfig->AddComponent<Artemis::MPMSimulatorConfigurator>();
            auto mpmContainerComponent = mpmGlobalConfig->AddComponent<Artemis::MPMParticleContainer>();
            mpmGlobalConfig->AddComponent<MPMTiming>();
            if (use3D)
            {
                mpmConfig->mPbmpmIterations = 5;
                mpmConfig->mMpmSubsteps     = 3;
                mpmConfig->mEnableRendering = false;
            }

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
            if (use3D)
            {
                cameraTransform->SetPosition({ 0.5f, 0.5f, -0.5f });
                camera->SetFov(60.0f);
                camera->SetCameraType(CameraType::Perspective);
            }

            auto defaultEmitter = node->AddGameObject("ParticleEmitter");
            auto emitter        = defaultEmitter->AddComponent<Artemis::MPMParticleEmitter>();
            if (GetApplicationState()->m_EditorMode)
            {
                emitter->SetEnable(false);
            }

            auto interactor   = node->AddGameObject("InteractiveControl");
            auto rigidEmitter = interactor->AddComponent<RigidEmitter>();
            rigidEmitter->SetEnable(false);
            interactor->AddComponent<MPMMouseInteractor>();

            if (use3D)
            {

                auto procMesh       = node->AddGameObject("MPMMesh");
                auto procMeshFilter = procMesh->AddComponent<MeshFilter>();
                procMeshFilter->SetMeshSource(mpmMesh);
                auto procMeshRenderer = procMesh->AddComponent<MeshRenderer>();
                procMeshRenderer->SetMaterialSource(material);
                auto procCfg = procMesh->AddComponent<MPMMeshingCfg>();

                auto planeGameObject = node->AddGameObject("Plane");
                auto planeFilter     = planeGameObject->AddComponent<MeshFilter>();
                planeFilter->SetMeshSource(plane);
                auto planeRenderer = planeGameObject->AddComponent<MeshRenderer>();
                planeRenderer->SetMaterialSource(material);
                auto planeTransform = planeGameObject->GetComponent<Transform>();
                planeTransform->SetScale({ 1.0f, 1.0f, 1.0f });
                planeTransform->SetPosition({ 0.5f, 0.0f, 0.5f });
            }

            m_sceneManager->SetActiveScene(scene);
        }

        void OnUpdate() override {}

        void OnEnd() override {}
    };
} // namespace Ifrit

int main()
{
    using namespace Ifrit;

    Ifrit::Reflection::RegisterReflectionTypes();

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
    Runtime::SetActiveApplication(&app);
    app.Run(info);
    return 0;
}
