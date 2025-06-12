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
    struct PushConstant_T
    {
        uint m_FrameIndex;
        TTexture2DHandle<float4> m_DirectLightingAtlas;
        TTexture2DHandle<float4> m_IndirectLightingAtlas;
        TTexture2DHandle<float4> m_AlbedoAtlas;
        TRWTexture2DHandle<float4> m_FinalLightingAtlas;
        uint m_CardResolution;
        uint m_CardAtlasResolution;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiSCDirectLightCardSizePerBlock,
        kAyanamiSCDirectLightCardSizePerBlock,
        kAyanamiSCDirectLightObjectsPerBlock
    )
    void SurfaceCacheCombineLightingCS(
        uint3 ThreadId : SV_DispatchThreadID
    )
    {
        uint MaxCardsInLine = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
        uint CardIndex_X = ThreadId.z % MaxCardsInLine;
        uint CardIndex_Y = ThreadId.z / MaxCardsInLine;
        
        uint2 CardOffset = uint2(CardIndex_X * PushConst.m_CardResolution, CardIndex_Y * PushConst.m_CardResolution);
        uint2 TileOffset = uint2(ThreadId.x, ThreadId.y);
        uint2 OverallOffset = CardOffset + TileOffset;

        float2 AtlasUV = (float2(OverallOffset) + 0.5f) / float2(PushConst.m_CardAtlasResolution, PushConst.m_CardAtlasResolution);

        float4 DirectLighting = PushConst.m_DirectLightingAtlas.SampleLevel(ESamplerType::SNearestClamp, AtlasUV,0);
        float4 IndirectLighting = PushConst.m_IndirectLightingAtlas.SampleLevel(ESamplerType::SNearestClamp, AtlasUV,0);
        float4 Albedo = PushConst.m_AlbedoAtlas.SampleLevel(ESamplerType::SNearestClamp, AtlasUV,0);

        float4 DiffuseLambertBRDF = Albedo / kPI;
        float4 FinalLighting = (DirectLighting + IndirectLighting) * DiffuseLambertBRDF;

        // temporal accumulation
        uint NumHistorySamples = (PushConst.m_FrameIndex > 0) ? PushConst.m_FrameIndex - 1 : 0; // Avoid negative index
        float4 PreviousColor = PushConst.m_FinalLightingAtlas.Load(OverallOffset);
        float4 MixedColor = lerp(PreviousColor, FinalLighting, 1.0f / (NumHistorySamples + 1));
        PushConst.m_FinalLightingAtlas.Store(OverallOffset, float4(MixedColor.rgb, 1.0f));
    }

    
}