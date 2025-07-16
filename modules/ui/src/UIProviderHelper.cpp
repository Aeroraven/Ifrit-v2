#include "ifrit/ui/UIProviderHelper.h"
#include "ifrit/ui/imgui/ImGuiProvider.h"
namespace Ifrit::UI
{
    IFRIT_APIDECL Owner<UIProvider> CreateUIProvider(EUIProviderType type)
    {
        switch (type)
        {
            case EUIProviderType::ImGui:
                return MakeOwner<ImGuiProvider>();
            default:
                iAssertion(false, "Unsupported UI provider type: {}", static_cast<int>(type));
                return nullptr;
        }
    }
} // namespace Ifrit::UI