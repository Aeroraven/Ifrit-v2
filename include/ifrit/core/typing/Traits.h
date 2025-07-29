
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/core/base/IfritBase.h"

namespace Ifrit
{
    template <typename T, typename... Types> using TypeIsAnyOf = std::disjunction<std::is_same<T, Types>...>;
    template <typename T, typename... Types> inline IF_CONSTEXPR bool TypeIsAnyOf_v = TypeIsAnyOf<T, Types...>::value;

    template <typename T> using TpTrivallyCopyable                      = std::is_trivially_copyable<T>;
    template <typename T> inline IF_CONSTEXPR bool TpTrivallyCopyable_v = TpTrivallyCopyable<T>::value;

    template <typename T> using TpTrivallyDestructible                      = std::is_trivially_destructible<T>;
    template <typename T> inline IF_CONSTEXPR bool TpTrivallyDestructible_v = TpTrivallyDestructible<T>::value;

    template <typename T> using TpTrivallyConstructible                      = std::is_trivially_constructible<T>;
    template <typename T> inline IF_CONSTEXPR bool TpTrivallyConstructible_v = TpTrivallyConstructible<T>::value;

    template <typename T, typename = void> struct TpIsIterable : std::false_type
    {
    };

    template <typename T>
    struct TpIsIterable<T,
        std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>())),
            decltype(*std::begin(std::declval<T>()))>> : std::true_type
    {
    };

    template <typename T> inline IF_CONSTEXPR bool TpIsIterable_v = TpIsIterable<T>::value;

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
    // Traits
    template <typename... Args> struct TTypeSet;
    template <typename T, typename U> struct TTraitIsAnyOf : std::false_type
    {
        static_assert(false, "TTraitIsAnyOf requires at least one type to compare against.");
    };
    template <typename T, typename... Types>
    struct TTraitIsAnyOf<T, TTypeSet<Types...>> : std::disjunction<std::is_same<T, Types>...>
    {
    };

    // Concepts

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

    template <typename T, typename U>
    concept IConceptIsAnyOf = TTraitIsAnyOf<T, U>::value;

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

} // namespace Ifrit