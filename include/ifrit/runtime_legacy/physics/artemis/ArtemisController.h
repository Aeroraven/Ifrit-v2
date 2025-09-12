#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/application/Subsystem.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"

namespace Ifrit::Runtime::Artemis
{
    enum class EPresetArtemisSimulator : u8
    {
        MPM
    };

    struct ArtemisControllerPrivate;
    class IFRIT_APIDECL ArtemisController : public ISubsystem
    {
    private:
        ArtemisControllerPrivate* m_Data;

    public:
        ArtemisController();
        ~ArtemisController();

        void                                   AddPresetSolver(EPresetArtemisSimulator preset);
        IArtemisSolver*                        GetPresetSolver(EPresetArtemisSimulator preset);
        void                                   SetTimestep(f32 timestep);

        virtual void                           OnInitialize(IApplication* app) override;
        virtual void                           OnShutdown() override;
        virtual void                           OnFrameBegin() override;
        virtual void                           OnFrameEnd() override;
        virtual Owner<RHI::RhiTaskSubmission>  OnPreRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual Owner<RHI::RhiTaskSubmission>  OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual void                           OnUpdate(Scene* scene) override;

        inline static Owner<ArtemisController> Create() { return MakeOwner<ArtemisController>(); }
    };
} // namespace Ifrit::Runtime::Artemis