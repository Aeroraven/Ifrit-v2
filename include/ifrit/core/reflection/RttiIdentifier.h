#pragma once
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/base/IfritBase.h"

namespace Ifrit::Reflection
{
    IFRIT_CORE_API u64 GetTypeIDHash(const std::type_info& typeInfo);
}
