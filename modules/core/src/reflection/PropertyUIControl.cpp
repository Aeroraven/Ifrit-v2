#include "ifrit/core/reflection/PropertyUIControl.h"
#include "ifrit/core/math/VectorDefs.h"
namespace Ifrit::Reflection
{
    template <typename T> IFRIT_APIDECL PropertyUIHandle<T>& GetPropertyUIHandleImpl()
    {
        static PropertyUIHandle<T> handle;
        return handle;
    }

    template IFRIT_APIDECL PropertyUIHandle<f32>& GetPropertyUIHandleImpl<f32>();
    template IFRIT_APIDECL PropertyUIHandle<i32>& GetPropertyUIHandleImpl<i32>();
    template IFRIT_APIDECL PropertyUIHandle<u32>& GetPropertyUIHandleImpl<u32>();
    template IFRIT_APIDECL PropertyUIHandle<f64>& GetPropertyUIHandleImpl<f64>();
    template IFRIT_APIDECL PropertyUIHandle<u64>& GetPropertyUIHandleImpl<u64>();
    template IFRIT_APIDECL PropertyUIHandle<i64>& GetPropertyUIHandleImpl<i64>();
    template IFRIT_APIDECL PropertyUIHandle<i8>& GetPropertyUIHandleImpl<i8>();
    template IFRIT_APIDECL PropertyUIHandle<u8>& GetPropertyUIHandleImpl<u8>();
    template IFRIT_APIDECL PropertyUIHandle<bool>& GetPropertyUIHandleImpl<bool>();
    template IFRIT_APIDECL PropertyUIHandle<Vector2f>& GetPropertyUIHandleImpl<Vector2f>();
    template IFRIT_APIDECL PropertyUIHandle<Vector3f>& GetPropertyUIHandleImpl<Vector3f>();
    template IFRIT_APIDECL PropertyUIHandle<Vector4f>& GetPropertyUIHandleImpl<Vector4f>();

    IFRIT_APIDECL PropertyAuxHandles&                  GetPropertyUIAuxHandles()
    {
        static PropertyAuxHandles handles;
        return handles;
    }

} // namespace Ifrit::Reflection
