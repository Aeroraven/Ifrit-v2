#include "ifrit/runtime/base/AssetReference.h"
#include "ifrit/runtime/base/EditorHandles.h"
namespace Ifrit::Runtime
{
    void AssetReferenceId::GetUIEditingHandle(const Reflection::PropertyMetadata& propMeta)
    {
        IF_LOG_DEBUG("Test", "GetUIEditingHandle called for AssetReferenceId with type: {}", (u32)mType);
        auto& handles = GetRuntimeEditorHandles();
        if (handles.AssetReferenceHandle)
        {
            handles.AssetReferenceHandle(*this);
        }
    }
} // namespace Ifrit::Runtime