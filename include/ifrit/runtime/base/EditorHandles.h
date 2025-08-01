#pragma once
#include "ifrit/runtime/base/AssetReference.h"
#include "ifrit/runtime/base/Base.h"
namespace Ifrit::Runtime
{

    struct RuntimeEditorHandles
    {
        Fn<void(AssetReferenceId&)> AssetReferenceHandle = nullptr;
    };

    IFRIT_RUNTIME_API RuntimeEditorHandles& GetRuntimeEditorHandles();
} // namespace Ifrit::Runtime