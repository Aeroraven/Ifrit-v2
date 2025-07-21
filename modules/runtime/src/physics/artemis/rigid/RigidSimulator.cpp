#include "ifrit/runtime/physics/artemis/rigid/RigidSimulator.h"
#include "ifrit.internal/runtime/physics/artemis/InternalConst.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/physics/artemis/ArtemisSceneData.h"

namespace Ifrit::Runtime::Artemis
{
    struct RigidSimulatorPrivateData
    {
        Scene* m_ActiveScene = nullptr;
    };

    IFRIT_APIDECL RigidSimulator::RigidSimulator() : m_Data(new RigidSimulatorPrivateData()) {}
    IFRIT_APIDECL RigidSimulator::~RigidSimulator()
    {
        delete m_Data;
        m_Data = nullptr;
    }

    IFRIT_APIDECL void             RigidSimulator::SetConfig(const RigidBaseConfig& cfg) { m_Config = cfg; }
    IFRIT_APIDECL RigidBaseConfig& RigidSimulator::GetActiveConfig() { return m_Config; }

    IFRIT_APIDECL void             RigidSimulator::CollectScene(Scene* scene)
    {
        m_Data->m_ActiveScene = scene;
        IF_LOG_ASSERTION("RigidSimulator",
            scene->GetPerFrameData()->m_ExtraData.count(Internal::kArtemisSceneDataKey) > 0,
            "ArtemisSceneData not found in scene's per-frame data");
    }

    IFRIT_APIDECL void RigidSimulator::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        // TODO
    }

} // namespace Ifrit::Runtime::Artemis