#pragma once
#include "ifrit/runtime/application/Subsystem.h"

namespace Ifrit::Runtime::Geometry
{
    struct ProceduralMeshUpdaterInternalData;
    class IFRIT_RUNTIME_API ProceduralMeshUpdater : public ISubsystem
    {
    public:
        ProceduralMeshUpdater();
        ~ProceduralMeshUpdater();

        virtual void                          OnInitialize(IApplication* app) override;
        virtual void                          OnShutdown() override;
        virtual void                          OnFrameBegin() override;
        virtual void                          OnFrameEnd() override;
        virtual Owner<RHI::RhiTaskSubmission> OnPreRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual Owner<RHI::RhiTaskSubmission> OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual void                          OnUpdate(Scene* scene) override;

    public:
        inline static Owner<ProceduralMeshUpdater> Create() { return MakeOwner<ProceduralMeshUpdater>(); }

    private:
        ProceduralMeshUpdaterInternalData* mData = nullptr;
    };

} // namespace Ifrit::Runtime::Geometry