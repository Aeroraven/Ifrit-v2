#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/core/base/IfritBase.h"

namespace Ifrit::Runtime
{
    enum class EPropertyEditorType
    {
        Range,
        Text,
        Select,
        Color
    };

    template <typename T> class PropertyEditorHandle
    {
    public:
        Fn<void(const char* name, T& value, T min, T max, T step)>         m_SliderCallback = nullptr;
        Fn<void(const char* name, T& value, Vec<Pair<T, String>> options)> m_SelectCallback = nullptr;
        Fn<void(const char* name, T& value)>                               m_TextCallback   = nullptr;
        Fn<void(const char* name, T& value)>                               m_ColorCallback  = nullptr;
    };

    struct PropertyEditorAxuHandles
    {
        Fn<void()> m_OnPreRegister  = nullptr;
        Fn<void()> m_OnPostRegister = nullptr;
    };

    template <typename T> IFRIT_RUNTIME_API PropertyEditorHandle<T>& GetPropertyEditorHandle();
    IFRIT_RUNTIME_API PropertyEditorAxuHandles&                      GetPropertyEditorAxuHandles();

    class ComponentPropertyBase
    {
    protected:
        const char* m_Name;
        Fn<void()>  m_EditorHandle = nullptr;

    public:
        ComponentPropertyBase(const char* name) : m_Name(name) {}
        virtual ~ComponentPropertyBase() = default;

        inline const char* GetName() const { return m_Name; }
        inline void        RegisterEditorHandle()
        {
            if (m_EditorHandle)
            {
                m_EditorHandle();
            }
        }
    };

    template <EPropertyEditorType E, typename T, typename = void> struct PropertyConstraint;

    template <typename T>
        requires std::is_arithmetic_v<T>
    struct PropertyConstraint<EPropertyEditorType::Range, T>
    {
        T m_Min;
        T m_Max;
        T m_Step;
        PropertyConstraint(T min, T max, T step = T(1)) : m_Min(min), m_Max(max), m_Step(step) {}
        IF_FORCEINLINE Fn<void()> GetEditorHandle(const char* name, T& value)
        {
            return [name, &value, m_Min = m_Min, m_Max = m_Max, m_Step = m_Step]() {
                auto handle = GetPropertyEditorHandle<T>();
                if (handle.m_SliderCallback)
                    handle.m_SliderCallback(name, value, m_Min, m_Max, m_Step);
            };
        }
    };

    template <typename T> struct PropertyConstraint<EPropertyEditorType::Select, T>
    {
        Vec<Pair<T, String>> m_Options;
        PropertyConstraint(Vec<Pair<T, String>> options) : m_Options(std::move(options)) {}
        IF_FORCEINLINE Fn<void()> GetEditorHandle(const char* name, T& value)
        {
            return [name, &value, options = m_Options]() {
                auto handle = GetPropertyEditorHandle<T>();
                if (handle.m_SelectCallback)
                    handle.m_SelectCallback(name, value, options);
            };
        }
    };

    template <> struct PropertyConstraint<EPropertyEditorType::Select, bool>
    {
        PropertyConstraint() {}
        IF_FORCEINLINE Fn<void()> GetEditorHandle(const char* name, bool& value)
        {
            return [name, &value]() {
                auto handle = GetPropertyEditorHandle<bool>();
                if (handle.m_SelectCallback)
                    handle.m_SelectCallback(name, value, { { true, "True" }, { false, "False" } });
            };
        }
    };

    template <typename T> struct PropertyConstraint<EPropertyEditorType::Text, T>
    {
        PropertyConstraint() {}
        IF_FORCEINLINE Fn<void()> GetEditorHandle(const char* name, T& value)
        {
            return [name, &value]() {
                auto handle = GetPropertyEditorHandle<T>();
                if (handle.m_TextCallback)
                    handle.m_TextCallback(name, value);
            };
        }
    };

    template <> struct PropertyConstraint<EPropertyEditorType::Color, Vector4f>
    {
        PropertyConstraint() {}
        IF_FORCEINLINE Fn<void()> GetEditorHandle(const char* name, Vector4f& value)
        {
            return [name, &value]() {
                auto handle = GetPropertyEditorHandle<Vector4f>();
                if (handle.m_ColorCallback)
                    handle.m_ColorCallback(name, value);
            };
        }
    };

    template <typename T, EPropertyEditorType E> class ComponentProperty : public ComponentPropertyBase
    {
    private:
        T& m_Value;

    public:
        ComponentProperty(const char* name, T& value, PropertyConstraint<E, T> constraint)
            : ComponentPropertyBase(name), m_Value(value)
        {
            m_EditorHandle = constraint.GetEditorHandle(name, m_Value);
        }
    };

} // namespace Ifrit::Runtime
