#include "ifrit/editor/EditorProvider.h"
#include "ifrit/runtime/base/Scene.h"

namespace Ifrit::Editor
{
    IFRIT_APIDECL void EditorProvider::OnInitialize(Runtime::IApplication* app) { m_Application = app; }
    IFRIT_APIDECL void EditorProvider::OnShutdown() { m_Application = nullptr; }
    IFRIT_APIDECL void EditorProvider::OnFrameBegin() {}
    IFRIT_APIDECL void EditorProvider::OnFrameEnd() {}
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> EditorProvider::OnPreRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }
    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> EditorProvider::OnPostRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }
    IFRIT_APIDECL void EditorProvider::OnUpdate(Runtime::Scene* scene) {}

} // namespace Ifrit::Editor
