#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/base/CoreBase.h"
namespace Ifrit::HAL
{

    struct FDynamicLinkedLibModule;

    IFRIT_CORE_API FDynamicLinkedLibModule* LoadDynamicLinkedLibrary(const String& path);
    IFRIT_CORE_API void*                    LoadDllFunction(FDynamicLinkedLibModule* lib, const String& functionName);
} // namespace Ifrit::HAL