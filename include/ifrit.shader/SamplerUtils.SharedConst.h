
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

// Some constants used for both cpp and shader code:

#ifndef IFRIT_SHADER_SAMPLER_UTILS_SHARED_CONST_H
#define IFRIT_SHADER_SAMPLER_UTILS_SHARED_CONST_H

#ifdef __cplusplus
    #include <cstdint>
    #include "ifrit/core/base/IfritBase.h"
    #define IFRIT_DEFINE_UINT(x, y) IF_CONSTEXPR u32 x = y;
    #define IFRIT_DEFINE_FLOAT(x, y) IF_CONSTEXPR float x = y;

namespace Ifrit::Runtime::SamplerUtils
{

#else
    #define IFRIT_DEFINE_UINT(x, y) const uint x = y
    #define IFRIT_DEFINE_FLOAT(x, y) const float x = y
#endif
    IFRIT_DEFINE_UINT(sLinearClamp, 0);
    IFRIT_DEFINE_UINT(sNearestClamp, 1);
    IFRIT_DEFINE_UINT(sLinearRepeat, 2);
    IFRIT_DEFINE_UINT(sNearestRepeat, 3);

#ifdef __cplusplus
} // namespace Ifrit::Runtime::Syaro
    #undef IFRIT_DEFINE_UINT
    #undef IFRIT_DEFINE_FLOAT
#endif
#endif