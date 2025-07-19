#include "ifrit/editor/EditorProviderHelper.h"
#include "ifrit/editor/imgui/ImGuiProvider.h"
namespace Ifrit::Editor
{
    IFRIT_APIDECL Owner<EditorProvider> CreateUIProvider(EEditorProviderType type)
    {
        switch (type)
        {
            case EEditorProviderType::ImGui:
                return MakeOwner<ImGuiProvider>();
            default:
                iAssertion(false, "Unsupported Editor provider type: {}", static_cast<int>(type));
                return nullptr;
        }
    }
} // namespace Ifrit::Editor
