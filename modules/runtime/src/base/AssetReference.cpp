#include "ifrit/runtime/base/AssetReference.h"
#include "ifrit/runtime/base/EditorHandles.h"
namespace Ifrit::Runtime
{

    IFRIT_APIDECL void AssetReferenceId::GetUIEditingHandle(const Reflection::PropertyMetadata& propMeta)
    {
        auto& handles = GetRuntimeEditorHandles();
        if (handles.AssetReferenceHandle)
        {
            handles.AssetReferenceHandle(propMeta.Name.c_str(), *this);
        }
    }
} // namespace Ifrit::Runtime