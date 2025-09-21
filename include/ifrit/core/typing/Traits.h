#pragma once
#include "ifrit/core/base/IfritBase.h"
#include <sstream>
#include <format>

namespace Ifrit
{
    // Traits
    template <class... T> inline constexpr bool TTraitAlwaysFalse = false;
    template <typename... Args> struct TTypeSet;

    template <typename T> struct TMemberType
    {
        using Type = T;
    };
    template <typename U, typename T> struct TMemberType<U T::*>
    {
        using Type      = U;
        using ClassType = T;
    };

    template <typename T> struct TMemberFunctionTrait : std::false_type
    {
    };

    template <typename R, typename C, typename... Args> struct TMemberFunctionTrait<R (C::*)(Args...)> : std::true_type
    {
        using ReturnType = R;
        using ClassType  = C;
        using ArgsTuple  = std::tuple<Args...>;
    };

    template <typename T, typename U> struct TTraitIsAnyOf : std::false_type
    {
        static_assert(TTraitAlwaysFalse<T>, "TTraitIsAnyOf requires at least one type to compare against.");
    };
    template <typename T, typename... Types>
    struct TTraitIsAnyOf<T, TTypeSet<Types...>> : std::disjunction<std::is_same<T, Types>...>
    {
    };

    template <typename T> struct TFunctionPointerType
    {
        using Type = T;
    };
    template <typename R, typename... Args> struct TFunctionPointerType<R (*)(Args...)>
    {
        using ReturnType = R;
        using ArgsTuple  = std::tuple<Args...>;
    };

    template <typename T, typename U> using TTraitIsSame                  = std::is_same<T, U>;
    template <typename T, typename U> using TTraitIsImplicitlyConvertible = std::is_convertible<T, U>;
    template <typename T, typename U> using TTraitIsConvertible           = std::is_convertible<T, U>;
    template <typename T> using TTraitDecayedType                         = std::decay_t<T>;
    template <bool B, typename T, typename F> using TTraitConditional     = std::conditional_t<B, T, F>;

    template <typename T> struct TTraitConstTypeRemoval
    {
        using Type = T;
    };
    template <typename T> struct TTraitConstTypeRemoval<const T>
    {
        using Type = T;
    };

    // Concepts

    template <typename T, typename... CtorArgs>
    concept IConceptIsConstructible = requires(CtorArgs&&... args) {
        { T(std::forward<CtorArgs>(args)...) } -> std::same_as<T>;
    };
    template <typename T, typename U>
    concept IConceptIsSame = TTraitIsSame<T, U>::value;

    template <typename T, typename U>
    concept IConceptIsAnyOf = TTraitIsAnyOf<T, U>::value;

    template <typename T>
    concept IConceptCustomSerializable = requires(T t) {
        { t.Serialize() } -> std::same_as<String>;
        { T::Deserialize(std::declval<String>()) } -> std::same_as<T>;
    };
    template <typename T>
    concept IConceptIsMemberPointer = requires(T t) {
        typename TMemberType<T>::Type;
        typename TMemberType<T>::ClassType;
    };
    template <typename T>
    concept IConceptIsMemberFunctionPointer = requires(T t) {
        typename TMemberFunctionTrait<T>::ReturnType;
        typename TMemberFunctionTrait<T>::ClassType;
        typename TMemberFunctionTrait<T>::ArgsTuple;
    };

    template <typename T>
    concept IConceptIsFunctionPointer = requires(T t) {
        typename TFunctionPointerType<T>::ReturnType;
        typename TFunctionPointerType<T>::ArgsTuple;
    };

    template <typename T>
    concept IConceptIsEnum = std::is_enum_v<T>;

    template <typename T, typename U>
    concept IConceptIsImplicitlyConvertible = TTraitIsImplicitlyConvertible<T, U>::value;

    template <typename T, typename U>
    concept IConceptIsDynamicallyConvertible = requires(T t) {
        { dynamic_cast<U>(t) } -> std::same_as<U>;
    };

#if IF_OPTION_DISCOURAGE_IMPLICIT_CONVERSION
    template <typename T, typename U>
    concept IConceptConversionGuarded = IConceptIsSame<T, U>;
#else
    template <typename T, typename U>
    concept IConceptConversionGuarded = IConceptIsImplicitlyConvertible<T, U> || IConceptIsSame<T, U>;
#endif

    template <typename T>
    concept IConceptIsEqualityComparable = requires(T t) {
        { t == t } -> std::same_as<bool>;
        { t != t } -> std::same_as<bool>;
    };

    template <typename T>
    concept IConceptIsHashable = requires(T t) {
        { std::hash<T>{}(t) } -> std::same_as<size_t>;
    };

    template <typename T>
    concept IConceptIsCopyable = std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T>;

    template <typename T>
    concept IConceptIsDestructible = std::is_destructible_v<T>;

    template <typename T>
    concept IConceptConvertibleToString = requires(T t, std::stringstream ss) {
        { ss << t } -> std::same_as<std::ostream&>;
    };

    template <typename T>
    concept IConceptConvertibleFromString = requires(T t, std::stringstream ss) {
        { ss >> t } -> std::same_as<std::stringstream&>;
    };
    template <typename T>
    concept IConceptIsScalar = std::is_scalar_v<T>;

    template <typename T>
    concept IConceptIsDecayed = IConceptIsSame<T, TTraitDecayedType<T>>;

    template <typename T, size_t Size>
    concept IConceptSizeofIs = sizeof(T) == Size;

    template <typename T>
    concept IConceptConvertibleToFuncPtr = IConceptIsFunctionPointer<T> || requires(T t) {
        { +t } -> IConceptIsFunctionPointer;
    };

    template <typename T>
    concept IConceptIsDefaultCopyable = std::is_trivially_copyable_v<T> && std::is_default_constructible_v<T>;

    // Concept Aliases
    template <typename T>
    concept IEnum = IConceptIsEnum<T>;

    template <typename T>
    concept IScalar = std::is_scalar_v<T>;

    template <typename T>
    concept IIntegral = std::is_integral_v<T>;

    template <typename T>
    concept IHashable = IConceptIsHashable<T>;

    template <typename T>
    concept IHashMapKey =
        IConceptIsHashable<T> && IConceptIsEqualityComparable<T> && IConceptIsCopyable<T> && IConceptIsDestructible<T>;

    template <typename T>
    concept IDefaultCopyable = IConceptIsDefaultCopyable<T>;

    template <typename T>
    concept IDecayed = IConceptIsDecayed<T>;

    // CountRef concepts
    template <typename T>
    concept IConceptCountReferable = requires(T t) {
        { t.AddRef() } -> std::same_as<void>;
        { t.Release() } -> std::same_as<void>;
        { static_cast<u32>(t.GetRefCount()) } -> std::same_as<u32>;
    };

    // Streaming concepts
    template <typename T>
    concept IConceptIsOutputStreamable = requires(std::ostream& os, T t) {
        { os << t } -> std::same_as<std::ostream&>;
    };

    template <typename T>
    concept IConceptIsInputStreamable = requires(std::istream& is, T t) {
        { is >> t } -> std::same_as<std::istream&>;
    };

    template <typename T>
    concept IConceptIsStreamable = IConceptIsOutputStreamable<T> && IConceptIsInputStreamable<T>;

    // Container Traits
    template <typename T> struct TTraitIsStlVector : std::false_type
    {
    };
    template <typename T> struct TTraitIsStlVector<std::vector<T>> : std::true_type
    {
    };
    template <typename T>
    concept IConceptIsStlVector = TTraitIsStlVector<T>::value;

    // Enum Type Traits

    template <typename T> struct TTraitUnderlyingType_Wrapper
    {
        using Type = T;
    };

    template <IConceptIsEnum T> struct TTraitUnderlyingType_Wrapper<T>
    {
        using Type = std::underlying_type_t<T>;
    };

    template <IConceptIsEnum T> using TTraitUnderlyingType = typename TTraitUnderlyingType_Wrapper<T>::Type;
    template <IConceptIsEnum T> using TEnumBitMask         = TTraitUnderlyingType<T>;

    template <typename T> using TEnumDecayedType = TTraitDecayedType<typename TTraitUnderlyingType_Wrapper<T>::Type>;

    // Argument Passing Helpers
    template <typename T>
    using TArgType =
        TTraitConditional<IConceptIsScalar<TTraitDecayedType<T>>, TTraitDecayedType<T>, const TTraitDecayedType<T>&>;

    template <IDecayed T> using TSinkArg  = T&&;
    template <IDecayed T> using TOutArg   = T&;
    template <IDecayed T> using TInOutArg = T&;

    // Formattable
    // https://stackoverflow.com/questions/72430369/how-to-check-that-a-type-is-formattable-using-type-traits-concepts
    template <typename T>
    concept IConceptIsFormattable =
        requires(T& v, std::format_context ctx) { std::formatter<std::remove_cvref_t<T>>().format(v, ctx); };

    template <typename T>
    concept IFormattable = IConceptIsFormattable<T>;

    template <typename... Args>
    concept IFormattableAll = (IFormattable<Args> && ...);

} // namespace Ifrit