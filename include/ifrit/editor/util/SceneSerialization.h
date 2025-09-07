#pragma once
#include "ifrit/editor/EditorBase.h"
#include "ifrit/runtime/base/Scene.h"
namespace Ifrit::Editor::Util
{
    IFRIT_EDITOR_API void ExportGameObjectAsPrefab(Runtime::GameObject* gameObject, const String& filePath);
}