#pragma once

#include "ifrit/ui/UIProvider.h"

namespace Ifrit::UI
{
    class IFRIT_UI_API ImGuiProvider : public UIProvider
    {
    protected:
        typedef UIProvider Super;

    public:
        virtual void                          OnInitialize(Runtime::IApplication* app) override;
        virtual void                          OnShutdown() override;
        virtual void                          OnFrameBegin() override;
        virtual void                          OnFrameEnd() override;
        virtual Owner<RHI::RhiTaskSubmission> OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual void                          OnUpdate(Runtime::Scene* scene) override;
    };
} // namespace Ifrit::UI
