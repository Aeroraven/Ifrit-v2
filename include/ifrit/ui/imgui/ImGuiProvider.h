#pragma once

#include "ifrit/ui/UIProvider.h"

namespace Ifrit::UI
{
    struct ImGuiProviderData;
    class IFRIT_UI_API ImGuiProvider : public UIProvider
    {
    protected:
        typedef UIProvider Super;
        ImGuiProviderData* m_Data = nullptr;

    public:
        ImGuiProvider();
        virtual ~ImGuiProvider();

        virtual void                          OnInitialize(Runtime::IApplication* app) override;
        virtual void                          OnShutdown() override;
        virtual void                          OnFrameBegin() override;
        virtual void                          OnFrameEnd() override;
        virtual Owner<RHI::RhiTaskSubmission> OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual void                          OnUpdate(Runtime::Scene* scene) override;
    };
} // namespace Ifrit::UI
