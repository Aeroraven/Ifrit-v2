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
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"

namespace IfritShader{

    struct HzbDataHandle
    {
        TRWStructuredBufferHandle<TRWTexture2DHandle<float>> m_HzbTexture;

        TRWTexture2DHandle<float> GetMip(uint MipLevel)
        {
            return m_HzbTexture.Load(MipLevel+1);
        }

        float GetPixel(uint MipLevel, uint2 UV)
        {
            uint2 UVInMips = UV >> MipLevel;
            return m_HzbTexture.Load(MipLevel+1).Load(UVInMips);
        }
    };

}