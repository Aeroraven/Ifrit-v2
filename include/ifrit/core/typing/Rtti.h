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
} // namespace Ifrit