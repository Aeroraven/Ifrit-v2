#include "ifrit/runtime/physics/artemis/ArtemisController.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMSimulator.h"
#include "ifrit/runtime/physics/artemis/ArtemisSimulator.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
namespace Ifrit::Runtime::Artemis
{
    struct ArtemisControllerPrivate
    {
        HashMap<EPresetArtemisSimulator, Owner<IArtemisSolver>> m_PresetSolvers;
        Owner<ArtemisSimulator>                                 m_ArtemisSim;
        u32                                                     m_FrameIdx = 0;
        f32                                                     m_Timestep = 1.0f / 60.0f;

    public:
        void AddPresetSolver(EPresetArtemisSimulator preset)
        {
            if (m_PresetSolvers.find(preset) == m_PresetSolvers.end())
            {
                Owner<MPMSimulator> p;
                switch (preset)
                {
                    case EPresetArtemisSimulator::MPM:
                        p = MakeOwner<MPMSimulator>();
                        p->SetDebugRenderTarget(GetActiveApplication()->GetDefaultColorImage());
                        m_PresetSolvers[preset] = std::move(p);
                        m_ArtemisSim->RegisterSolver(m_PresetSolvers[preset].get());
                        break;
                    default:
                        IF_LOG_ERROR("Artemis", "Unknown preset Artemis simulator type.");
                        break;
                }
            }
        }

        IArtemisSolver* GetPresetSolver(EPresetArtemisSimulator preset)
        {
            auto it = m_PresetSolvers.find(preset);
            if (it != m_PresetSolvers.end())
            {
                return it->second.get();
            }
            return nullptr;
        }
    };

    IFRIT_APIDECL ArtemisController::ArtemisController() : m_Data(new ArtemisControllerPrivate())
    {
        m_Data->m_ArtemisSim = MakeOwner<ArtemisSimulator>(GetActiveApplication());
    }

    IFRIT_APIDECL      ArtemisController::~ArtemisController() { delete m_Data; }

    IFRIT_APIDECL void ArtemisController::AddPresetSolver(EPresetArtemisSimulator preset)
    {
        m_Data->AddPresetSolver(preset);
    }

    IFRIT_APIDECL void            ArtemisController::SetTimestep(f32 timestep) { m_Data->m_Timestep = timestep; }

    IFRIT_APIDECL IArtemisSolver* ArtemisController::GetPresetSolver(EPresetArtemisSimulator preset)
    {
        return m_Data->GetPresetSolver(preset);
    }

    IFRIT_APIDECL void ArtemisController::OnInitialize(IApplication* app) {}
    IFRIT_APIDECL void ArtemisController::OnShutdown() {}
    IFRIT_APIDECL void ArtemisController::OnFrameBegin() { m_Data->m_FrameIdx = (m_Data->m_FrameIdx + 1) % 2; }
    IFRIT_APIDECL void ArtemisController::OnFrameEnd() {}
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> ArtemisController::OnPreRendering(
        RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> ArtemisController::OnPostRendering(
        RHI::RhiTaskSubmission* prevSubmission)
    {
        return m_Data->m_ArtemisSim->Update(m_Data->m_Timestep, { prevSubmission });
    }
    IFRIT_APIDECL void ArtemisController::OnUpdate(Scene* scene)
    {
        m_Data->m_ArtemisSim->CollectScene(scene, m_Data->m_FrameIdx);
    }

} // namespace Ifrit::Runtime::Artemis