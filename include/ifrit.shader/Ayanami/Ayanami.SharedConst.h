
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

#ifndef AYANAMI_SHADER_SHARED_CONST_H
#define AYANAMI_SHADER_SHARED_CONST_H

#ifdef __cplusplus
    #include <cstdint>
    #include "ifrit/common/base/IfritBase.h"
    #define AYANAMI_DEFINE_UINT(x, y) IF_CONSTEXPR uint32_t x = y;
    #define AYANAMI_DEFINE_FLOAT(x, y) IF_CONSTEXPR float x = y;

namespace Ifrit::Core::Ayanami::Config
{

#else
    #define AYANAMI_DEFINE_UINT(x, y) const uint x = y
    #define AYANAMI_DEFINE_FLOAT(x, y) const float x = y
#endif

// SDF expanding. This is used to expand the SDF volume to avoid artifacts in thin geometry.
#define AYANAMI_ENABLE_SDF_EXPAND 1

    // Constants for Ayanami
    AYANAMI_DEFINE_FLOAT(kAyanamiSDFExpandRatio, 0.1f);

    AYANAMI_DEFINE_UINT(kAyanamiGlobalDFCompositeTileSize, 8);
    AYANAMI_DEFINE_UINT(kAyanamiGlobalDFRayMarchTileSize, 16);

    AYANAMI_DEFINE_UINT(kAyanamiRadianceInjectionObjectsPerBlock, 8);
    AYANAMI_DEFINE_UINT(kAyanamiRadianceInjectionCardSizePerBlock, 8);

#ifdef __cplusplus
} // namespace Ifrit::Core::AYANAMI
    #undef AYANAMI_DEFINE_UINT
    #undef AYANAMI_DEFINE_FLOAT
#endif

#endif