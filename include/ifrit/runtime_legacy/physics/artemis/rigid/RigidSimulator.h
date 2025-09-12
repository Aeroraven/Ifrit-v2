#pragma once
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/runtime/forwarding/FwdScene.h"
#include "ifrit/runtime/physics/artemis/rigid/RigidBase.h"

namespace Ifrit::Runtime::Artemis
{

    struct RigidSimulatorPrivateData;

    class IFRIT_RUNTIME_API RigidSimulator : public IArtemisSolver
    {
    public:
        RigidSimulator();
        ~RigidSimulator();

        void             SetConfig(const RigidBaseConfig& cfg);
        RigidBaseConfig& GetActiveConfig();

        void             CollectScene(Scene* scene) override;
        virtual void     RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime) override;

    private:
        RigidSimulatorPrivateData* m_Data = nullptr;
        RigidBaseConfig            m_Config;
    };
} // namespace Ifrit::Runtime::Artemis
