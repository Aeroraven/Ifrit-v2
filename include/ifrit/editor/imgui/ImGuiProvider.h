#pragma once

#include "ifrit/editor/EditorProvider.h"

namespace Ifrit::Editor
{
    struct ImGuiProviderData;
    class IFRIT_EDITOR_API ImGuiProvider : public EditorProvider
    {
    protected:
        typedef EditorProvider Super;
        ImGuiProviderData*     m_Data = nullptr;

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
} // namespace Ifrit::Editor
