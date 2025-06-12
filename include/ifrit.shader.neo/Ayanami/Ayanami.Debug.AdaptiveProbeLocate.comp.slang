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


#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedConst.hlsli"

namespace IfritShader
{

    struct AdaptiveProbeElement
    {
        uint m_PackedCoord;
    };

    struct AdaptiveProbeCounter
    {
        uint m_Counter;
        uint m_InvoX;
        uint m_InvoY;
        uint m_InvoZ;
    };

    struct PushConstant_T
    {
        TRWStructuredBufferHandle<AdaptiveProbeCounter> m_ProbeCounter;
        TRWStructuredBufferHandle<AdaptiveProbeElement> m_ProbeElement;
        TRWTexture2DHandle<float4> m_OutputTex;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    uint2 UnpackLocation(uint PackedCoords){
        uint y = PackedCoords & 0xFFFF;
        uint x = (PackedCoords >> 16) & 0xFFFF;
        return uint2(x,y);
    }


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiScrProbeAdaptiveGroupKernelSize,
        1,
        1
    )
    void DebugAdaptiveProbeLocateCS(
        uint3 ThreadId : SV_DispatchThreadID
    )
    {
        uint ProbeIndex = ThreadId.x;
        if (ProbeIndex >= PushConst.m_ProbeCounter.Load().m_Counter)
        {
            return; // Out of bounds
        }

        AdaptiveProbeElement probeElement = PushConst.m_ProbeElement.Load(ProbeIndex);
        uint2 Coords = UnpackLocation(probeElement.m_PackedCoord);
        PushConst.m_OutputTex.Store(Coords, float4(0.0f, 1.0f, 0.0f, 1.0f)); // Store green color at the probe location
    }
}