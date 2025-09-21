
#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/typing/Traits.h"

namespace Ifrit
{

    inline Vec<String> SplitString(const String& str, const String& delimiter)
    {
        Vec<String> result;
        size_t      start = 0;
        size_t      end   = str.find(delimiter);
        while (end != String::npos)
        {
            result.push_back(str.substr(start, end - start));
            start = end + delimiter.length();
            end   = str.find(delimiter, start);
        }
        result.push_back(str.substr(start, end));
        return result;
    }

    inline String JoinString(const Vec<String>& strings, const String& delimiter)
    {
        String result;
        for (size_t i = 0; i < strings.size(); ++i)
        {
            result += strings[i];
            if (i < strings.size() - 1)
            {
                result += delimiter;
            }
        }
        return result;
    }

    template <IConceptConvertibleToString T>
        requires IConceptIsScalar<TTraitDecayedType<T>>
    IF_NODISCARD IF_FORCEINLINE constexpr String ToString(T value) noexcept
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

    template <IConceptConvertibleToString T>
        requires(!IConceptIsScalar<TTraitDecayedType<T>>)
    IF_NODISCARD IF_FORCEINLINE constexpr String ToString(const T& value) noexcept
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }

} // namespace Ifrit