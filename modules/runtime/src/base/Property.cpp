#include "ifrit/runtime/base/Property.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime
{
    template <typename T> IFRIT_APIDECL PropertyEditorHandle<T>& GetPropertyEditorHandle()
    {
        static PropertyEditorHandle<T> handle;
        return handle;
    }

    template IFRIT_APIDECL PropertyEditorHandle<f32>& GetPropertyEditorHandle<f32>();
    template IFRIT_APIDECL PropertyEditorHandle<i32>& GetPropertyEditorHandle<i32>();

} // namespace Ifrit::Runtime