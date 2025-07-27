#include "ifrit/editor/EditorProviderHelper.h"
#include "ifrit/editor/imgui/ImGuiProvider.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Editor
{
    IFRIT_APIDECL Owner<EditorProvider> CreateEditorProvider(EEditorProviderType type)
    {
        switch (type)
        {
            case EEditorProviderType::ImGui:
                return MakeOwner<ImGuiProvider>();
            default:
                IF_LOG_CRITICAL("EditorProviderHelper", "Unsupported Editor provider type: {}", static_cast<int>(type));
                return nullptr;
        }
    }
} // namespace Ifrit::Editor
