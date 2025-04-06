
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
#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/imaging/base/Base.h"
#include "ifrit/core/algo/SizedBuffer.h"

namespace Ifrit::Imaging::Compress
{
    enum class TextureFormat
    {
        RGBA8_UNORM,
        RG8_UNORM,
        ATSC_6x6_UNORM,
        BC7_UNORM,
    };

    enum class CompressionAlgo
    {
        BC5,
        BC7
    };

    IFRIT_IMAGING_API void DiscardBAChannel(
        const RSizedBuffer& in, RSizedBuffer& out, u32 width, u32 height, u32 depth, u32 inChannels, u32 channelWidth);

    IFRIT_IMAGING_API void WriteTex2DToBlockCompressedFile(const RSizedBuffer& in, const String& outFile,
        TextureFormat fmt, u32 baseWidth, u32 baseHeight, u32 baseDepth, CompressionAlgo algo);

    IFRIT_IMAGING_API void ReadBlockCompressedTex2DFromFile(
        RSizedBuffer& out, const String& inFile, u32& baseWidth, u32& baseHeight, u32& baseDepth);

} // namespace Ifrit::Imaging::Compress