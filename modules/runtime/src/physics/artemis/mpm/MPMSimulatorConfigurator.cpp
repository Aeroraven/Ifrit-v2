#include "ifrit/runtime/physics/artemis/mpm/MPMSimulatorConfigurator.h"

namespace Ifrit::Runtime::Artemis
{
    struct MPMSimulatorConfiguratorPrivateData
    {
        MPMSimulator* m_ActiveSimulator = nullptr;
    };

    IFRIT_APIDECL MPMSimulatorConfigurator::MPMSimulatorConfigurator()
        : Component(), m_Data(new MPMSimulatorConfiguratorPrivateData())
    {
    }

    IFRIT_APIDECL MPMSimulatorConfigurator::MPMSimulatorConfigurator(GameObject* owner)
        : Component(owner), m_Data(new MPMSimulatorConfiguratorPrivateData())
    {
    }

    IFRIT_APIDECL MPMSimulatorConfigurator::~MPMSimulatorConfigurator()
    {
        delete m_Data;
        m_Data = nullptr;
    }

    IFRIT_APIDECL void MPMSimulatorConfigurator::SetActiveSimulator(MPMSimulator* sim)
    {
        m_Data->m_ActiveSimulator = sim;
    }

    IFRIT_APIDECL void MPMSimulatorConfigurator::OnUpdate()
    {
        if (m_Data->m_ActiveSimulator)
        {
            auto activeCfg                                        = m_Data->m_ActiveSimulator->GetActiveConfig();
            activeCfg.m_Substeps                                  = mMpmSubsteps;
            activeCfg.m_PbMpmIterations                           = mPbmpmIterations;
            activeCfg.m_PbMpmDefaultElasticityInterpolationFactor = mPbmpmElasticityRatio;
            activeCfg.m_PbMpmDefaultElasticityRelaxationFactor    = mPbmpmElasticityRelax;
            activeCfg.m_Gravity                                   = mGravity;
            activeCfg.m_EnableRigidCoupling                       = mEnableRigidCoupling;
            m_Data->m_ActiveSimulator->SetConfig(activeCfg);
            m_Data->m_ActiveSimulator->SetDefaultSize(mPointSize);
        }
        if (mClearParticles)
        {
            if (m_Data->m_ActiveSimulator)
            {
                m_Data->m_ActiveSimulator->RequestClearParticles();
            }
            mClearParticles = false; // Reset the flag after clearing
        }
    }

} // namespace Ifrit::Runtime::Artemis
