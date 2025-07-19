#pragma once
#include "ifrit/editor/EditorProvider.h"

namespace Ifrit::Editor
{
    enum class EEditorProviderType
    {
        ImGui, // ImGui provider
    };

    IFRIT_EDITOR_API Owner<EditorProvider> CreateEditorProvider(EEditorProviderType type);
} // namespace Ifrit::Editor
