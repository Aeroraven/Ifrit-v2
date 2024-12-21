
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

#ifndef IFRIT_FFTCONV2D_SHADER_SHARED_CONST_H
#define IFRIT_FFTCONV2D_SHADER_SHARED_CONST_H

#ifdef __cplusplus
#include <cstdint>
#define SYARO_DEFINE_UINT(x, y) constexpr uint32_t x = y;
#define SYARO_DEFINE_FLOAT(x, y) constexpr float x = y;

namespace Ifrit::Core::FFTConv2DConfig {

#else
#define SYARO_DEFINE_UINT(x, y) const uint x = y
#define SYARO_DEFINE_FLOAT(x, y) const float x = y
#endif

SYARO_DEFINE_UINT(cMaxSupportedTextureSize, 1024);
SYARO_DEFINE_UINT(cMaxSupportedTextureSizeLog2, 10);
SYARO_DEFINE_UINT(cMaxSupportedTextureSizeHalf, 512);
SYARO_DEFINE_UINT(cThreadGroupSizeX, 512);

#ifdef __cplusplus
}
#endif

#endif