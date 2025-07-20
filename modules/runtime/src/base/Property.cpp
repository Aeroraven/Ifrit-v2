#include "ifrit/runtime/base/Property.h"
#include "ifrit/runtime/base/Base.h"
#include "ifrit/core/math/VectorDefs.h"

namespace Ifrit::Runtime
{
    template <typename T> IFRIT_APIDECL PropertyEditorHandle<T>& GetPropertyEditorHandle()
    {
        static PropertyEditorHandle<T> handle;
        return handle;
    }

    template IFRIT_APIDECL PropertyEditorHandle<f32>& GetPropertyEditorHandle<f32>();
    template IFRIT_APIDECL PropertyEditorHandle<i32>& GetPropertyEditorHandle<i32>();
    template IFRIT_APIDECL PropertyEditorHandle<u32>& GetPropertyEditorHandle<u32>();
    template IFRIT_APIDECL PropertyEditorHandle<i8>& GetPropertyEditorHandle<i8>();
    template IFRIT_APIDECL PropertyEditorHandle<u8>& GetPropertyEditorHandle<u8>();
    template IFRIT_APIDECL PropertyEditorHandle<bool>& GetPropertyEditorHandle<bool>();
    template IFRIT_APIDECL PropertyEditorHandle<Vector2f>& GetPropertyEditorHandle<Vector2f>();
    template IFRIT_APIDECL PropertyEditorHandle<Vector3f>& GetPropertyEditorHandle<Vector3f>();
    template IFRIT_APIDECL PropertyEditorHandle<Vector3f>& GetPropertyEditorHandle<Vector3f>();

    IFRIT_APIDECL PropertyEditorAxuHandles&                GetPropertyEditorAxuHandles()
    {
        static PropertyEditorAxuHandles handles;
        return handles;
    }

} // namespace Ifrit::Runtime
