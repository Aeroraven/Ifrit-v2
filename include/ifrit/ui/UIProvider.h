#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/ui/UIBase.h"
#include "ifrit/runtime/application/Subsystem.h"

namespace Ifrit::UI
{
    class IFRIT_UI_API UIProvider : public Ifrit::Runtime::ISubsystem
    {
    protected:
        Runtime::IApplication* m_Application = nullptr;

    public:
        virtual void                          OnInitialize(Runtime::IApplication* app) override;
        virtual void                          OnShutdown() override;
        virtual void                          OnFrameBegin() override;
        virtual void                          OnFrameEnd() override;
        virtual Owner<RHI::RhiTaskSubmission> OnPreRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual Owner<RHI::RhiTaskSubmission> OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual void                          OnUpdate(Runtime::Scene* scene) override;
    };
} // namespace Ifrit::UI
