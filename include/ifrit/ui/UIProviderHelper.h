#pragma once
#include "ifrit/ui/UIProvider.h"

namespace Ifrit::UI
{
    enum class EUIProviderType
    {
        ImGui, // ImGui provider
    };

    IFRIT_UI_API Owner<UIProvider> CreateUIProvider(EUIProviderType type);
} // namespace Ifrit::UI