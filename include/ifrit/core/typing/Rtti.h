#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"

namespace Ifrit
{
    template <typename T> String GetDynamicTypeName(T* ptr)
    {
#ifdef _MSC_VER
        String name = typeid(*ptr).name();
        if (name.rfind("class ") == 0)
        {
            name = name.substr(6);
        }
        else if (name.rfind("struct ") == 0)
        {
            name = name.substr(7);
        }
        return name;
#else
        return typeid(*ptr).name();
#endif
    }

    template <typename T> String GetDynamicTypeNameWithoutNamespace(T* ptr)
    {
        String name = GetDynamicTypeName(ptr);
        auto   pos  = name.find_last_of("::");
        if (pos != String::npos)
        {
            return name.substr(pos + 1);
        }
        return name;
    }

    template <typename T> String GetDynamicTypeNamespace(T* ptr)
    {
        String name = GetDynamicTypeName(ptr);
        auto   pos  = name.find_last_of("::");
        if (pos != String::npos)
        {
            return name.substr(0, std::max(static_cast<decltype(pos)>(0), pos - 1));
        }
        return "";
    }
} // namespace Ifrit
