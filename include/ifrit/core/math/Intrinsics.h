#pragma once
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/base/IfritBase.h"
#ifdef _MSC_VER
    #include <intrin.h>
#endif

namespace Ifrit::Math
{
    inline u64 IntegerPack32(u32 x, u32 y) { return (u64)x | ((u64)y << 32); }
    inline u32 IntegerUnpack32From64First(u64 x) { return (u32)x; }
    inline u32 IntegerUnpack32From64Second(u64 x) { return (u32)(x >> 32); }

} // namespace Ifrit::Math