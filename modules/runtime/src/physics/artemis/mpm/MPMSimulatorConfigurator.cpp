#include "ifrit/runtime/physics/artemis/mpm/MPMSimulatorConfigurator.h"

namespace Ifrit::Runtime::Artemis
{
    struct MPMSimulatorConfiguratorPrivateData
    {
        MPMSimulator* m_ActiveSimulator      = nullptr;
        i32           m_MpmSubsteps          = 3;
        i32           m_PbMpmIterations      = 8;
        f32           m_PbMpmElasticityRatio = 0.5f;
        f32           m_PbMpmElasticityRelax = 1.5f;
        Vector3f      m_Gravity              = Vector3f(0.0f, -1.0f, 0.0f);
    };

    IFRIT_APIDECL MPMSimulatorConfigurator::MPMSimulatorConfigurator()
        : Component(), m_Data(new MPMSimulatorConfiguratorPrivateData())
    {
    }

    IFRIT_APIDECL MPMSimulatorConfigurator::MPMSimulatorConfigurator(Ref<GameObject> owner)
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
            activeCfg.m_Substeps                                  = m_Data->m_MpmSubsteps;
            activeCfg.m_PbMpmIterations                           = m_Data->m_PbMpmIterations;
            activeCfg.m_PbMpmDefaultElasticityInterpolationFactor = m_Data->m_PbMpmElasticityRatio;
            activeCfg.m_PbMpmDefaultElasticityRelaxationFactor    = m_Data->m_PbMpmElasticityRelax;
            activeCfg.m_Gravity                                   = m_Data->m_Gravity;
            m_Data->m_ActiveSimulator->SetConfig(activeCfg);
        }
    }

    IFRIT_APIDECL void MPMSimulatorConfigurator::SetupProperties()
    {
        AddProperty<i32, EPropertyEditorType::Range>("MPM Substeps", m_Data->m_MpmSubsteps, 1, 50, 1);
        AddProperty<i32, EPropertyEditorType::Range>("PbMpm Iters", m_Data->m_PbMpmIterations, 2, 100, 1);
        AddProperty<f32, EPropertyEditorType::Range>(
            "PbMpm Rigidity", m_Data->m_PbMpmElasticityRatio, 0.0f, 1.0f, 0.01f);
        AddProperty<f32, EPropertyEditorType::Range>(
            "PbMpm Elastic Rel.", m_Data->m_PbMpmElasticityRelax, 0.0f, 10.0f, 0.01f);
        AddProperty<Vector3f, EPropertyEditorType::Text>("Gravity", m_Data->m_Gravity);
    }
} // namespace Ifrit::Runtime::Artemis