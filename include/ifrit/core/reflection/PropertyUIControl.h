#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/math/VectorDefs.h"
#include "ifrit/core/reflection/PropertyMeta.h"
#include "ifrit/core/reflection/Object.h"
#include "ifrit/core/typing/EnumReflection.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Reflection
{

    template <typename T>
    concept IConceptEditableType =
        TTraitIsAnyOf<T, TTypeSet<f32, i32, u32, f64, u64, i64, i8, u8, bool, Vector2f, Vector3f, Vector4f>>::value;

    template <typename T>
    concept IConceptHasCustomEditingHandle = requires(T t) {
        {
            t.GetUIEditingHandle(std::declval<const PropertyMetadata&>())
        } -> std::same_as<void>;
    };

    template <typename T> class PropertyUIHandle
    {
    public:
        Fn<void(const char* name, T& value, T min, T max, Fn<bool()> predicate)> mSliderCallback = nullptr;
        Fn<void(const char* name, T& value, Vec<Pair<T, String>> options, Fn<bool()> predicate)> mSelectCallback =
            nullptr;
        Fn<void(const char* name, T& value, Fn<bool()> predicate)> mTextCallback  = nullptr;
        Fn<void(const char* name, T& value, Fn<bool()> predicate)> mColorCallback = nullptr;
    };

    class FunctionUIHandle
    {
    public:
        Fn<void(const char* name, Fn<void()>)> mFunctionCallback = nullptr;
    };

    struct PropertyAuxHandles
    {
        Fn<void()> m_OnPreRegister  = nullptr;
        Fn<void()> m_OnPostRegister = nullptr;
    };

    template <typename T> IFRIT_CORE_API PropertyUIHandle<T>& GetPropertyUIHandleImpl();
    IFRIT_CORE_API FunctionUIHandle&                          GetFunctionUIHandle();
    IFRIT_CORE_API PropertyAuxHandles&                        GetPropertyUIAuxHandles();

    template <typename T>
        requires IConceptEditableType<T>
    PropertyUIHandle<T>& GetPropertyUIHandle()
    {
        return GetPropertyUIHandleImpl<T>();
    }

    template <typename T>
        requires std::is_same_v<bool, T> || std::is_enum_v<T>
    void ProcessPropertyMetadataSelect(const PropertyMetadata& metadata, Object& propObj)
    {
        bool visible  = metadata.Editable != EPropertyEditable::None;
        bool editable = metadata.Editable == EPropertyEditable::Editable;
        T&   valueRef = propObj.As<T>();
        if constexpr (std::is_same_v<T, bool>)
        {
            auto handles = GetPropertyUIHandle<bool>();
            if (handles.mSelectCallback)
            {
                handles.mSelectCallback(metadata.Name.c_str(), valueRef,
                    Vec<Pair<bool, String>>{ { true, "True" }, { false, "False" } }, [editable]() { return editable; });
            }
        }
        else if constexpr (std::is_enum_v<T>)
        {
            using U                         = std::underlying_type_t<T>;
            auto                 handles    = GetPropertyUIHandle<U>();
            auto                 enumValues = GetEnumValues<T>();
            auto                 enumNames  = GetEnumNames<T>();
            Vec<Pair<U, String>> options;
            options.reserve(enumValues.size());
            for (size_t i = 0; i < enumValues.size(); ++i)
            {
                Pair<U, String> w;
                w.first  = GetEnumUnderlyingValue(enumValues[i]);
                w.second = String{ enumNames[i] };
                options.push_back(w);
            }
            if (handles.mSelectCallback)
            {
                handles.mSelectCallback(
                    metadata.Name.c_str(), reinterpret_cast<U&>(valueRef), options, [editable]() { return editable; });
            }
        }
        else
        {
            static_assert(IConceptEditableType<T>, "Unsupported type for property UI control.");
        }
    }

    template <typename T>
        requires(IConceptEditableType<T> || std::is_enum_v<T>)
    void ProcessPropertyMetadata(const PropertyMetadata& metadata, Object& propObj)
    {
        bool visible  = metadata.Editable != EPropertyEditable::None;
        bool editable = metadata.Editable == EPropertyEditable::Editable;
        if (visible)
        {
            T& valueRef = propObj.As<T>();

            if (metadata.UIControl == EPropertyUIControl::UISelect)
            {
                if constexpr (std::is_same_v<bool, T> || std::is_enum_v<T>)
                {
                    ProcessPropertyMetadataSelect<T>(metadata, propObj);
                }
            }
            else
            {
                if constexpr (IConceptEditableType<T>)
                {
                    auto handles = GetPropertyUIHandle<T>();
                    switch (metadata.UIControl)
                    {
                        case EPropertyUIControl::UISlider:
                            if (handles.mSliderCallback)
                            {
                                // TODO: int ranges might be influenced by fp precision.
                                if constexpr (!(std::is_same_v<T, Vector4f> || std::is_same_v<T, Vector3f>
                                                  || std::is_same_v<T, Vector2f>))
                                {
                                    handles.mSliderCallback(metadata.Name.c_str(), valueRef,
                                        static_cast<T>(metadata.UIClampMin), static_cast<T>(metadata.UIClampMax),
                                        [editable]() { return editable; });
                                }
                            }
                            break;
                        case EPropertyUIControl::UIText:
                            if (handles.mTextCallback)
                            {
                                handles.mTextCallback(
                                    metadata.Name.c_str(), valueRef, [editable]() { return editable; });
                            }
                            break;
                        case EPropertyUIControl::UIColor:
                            if (handles.mColorCallback)
                            {
                                handles.mColorCallback(
                                    metadata.Name.c_str(), valueRef, [editable]() { return editable; });
                            }
                            break;
                        default:
                            IF_LOG_CRITICAL(
                                "Reflector", "Unsupported UI control type: {}", static_cast<int>(metadata.UIControl));
                            break;
                    }
                }
            }
        }
    }

} // namespace Ifrit::Reflection
