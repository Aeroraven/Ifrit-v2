#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() Prefab
    {
    public:
        IF_PROPERTY()
        String mSerializedData;
    };
} // namespace Ifrit::Runtime
