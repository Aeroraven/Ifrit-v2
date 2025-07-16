#include "ifrit/ui/UIProvider.h"
#include "ifrit/runtime/base/Scene.h"

namespace Ifrit::UI
{
    IFRIT_APIDECL void UIProvider::OnInitialize(Runtime::IApplication* app) { m_Application = app; }
    IFRIT_APIDECL void UIProvider::OnShutdown() { m_Application = nullptr; }
    IFRIT_APIDECL void UIProvider::OnFrameBegin() {}
    IFRIT_APIDECL void UIProvider::OnFrameEnd() {}
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> UIProvider::OnPreRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> UIProvider::OnPostRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }
    IFRIT_APIDECL void UIProvider::OnUpdate(Runtime::Scene* scene) {}

} // namespace Ifrit::UI
