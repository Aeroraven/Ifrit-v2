#include "ifrit/runtime/base/EditorHandles.h"

namespace Ifrit::Runtime
{
    IFRIT_RUNTIME_API RuntimeEditorHandles& GetRuntimeEditorHandles()
    {
        static RuntimeEditorHandles handles;
        return handles;
    }
} // namespace Ifrit::Runtime