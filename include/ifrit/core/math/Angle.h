#pragma once
#include "ifrit/core/base/IfritBase.h"

namespace Ifrit::Math
{
    f32 AngleToRadian(f32 angle) { return angle * (3.14159265358979323846f / 180.0f); }
} // namespace Ifrit::Math