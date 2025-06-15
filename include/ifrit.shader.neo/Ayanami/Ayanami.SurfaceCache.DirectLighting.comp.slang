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
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"

namespace IfritShader {
namespace Ayanami {

    struct PushConstant_T
    {
        float4 m_LightDir; // todo
        TRWTexture2DHandle<float4> m_DirectLighting;
        TTexture2DHandle<float4> m_NormalAtlas;
        TTexture2DHandle<float> m_ShadowMask;
        uint m_CardResolution;
        uint m_CardAtlasResolution;
        TRWStructuredBufferHandle<FMeshDFDescriptor> m_MeshDFDesc;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(
        kAyanamiSCDirectLightCardSizePerBlock,
        kAyanamiSCDirectLightCardSizePerBlock,
        kAyanamiSCDirectLightObjectsPerBlock
    )
    void SurfaceCacheDirectLightingCS(uint3 ThreadID : SV_DispatchThreadID,
                                        uint3 GroupID : SV_GroupID,
                                        uint3 GroupThreadID : SV_GroupThreadID)
    {
        uint MaxCardsInLine = PushConst.m_CardAtlasResolution / PushConst.m_CardResolution;
        uint CardIndex_X = ThreadID.z % MaxCardsInLine;
        uint CardIndex_Y = ThreadID.z / MaxCardsInLine;
        uint2 CardOffset = uint2(CardIndex_X * PushConst.m_CardResolution, CardIndex_Y * PushConst.m_CardResolution);
        uint2 TileOffset = uint2(ThreadID.x, ThreadID.y);
        uint2 OverallOffset = CardOffset + TileOffset;


        float2 AtlasUV = (float2(OverallOffset) + float2(0.5)) / float2(PushConst.m_CardAtlasResolution);

        float3 NormalLocal = PushConst.m_NormalAtlas.SampleLevel(ESamplerType::SLinearClamp, AtlasUV, 0.0f).xyz;
        NormalLocal = NormalLocal * 2.0f - 1.0f; 

        uint MeshId = ThreadID.z / 6;
        FMeshDFDescriptor MeshDesc = PushConst.m_MeshDFDesc.Load(MeshId);
        float4x4 LocalToWorld = MeshDesc.m_Transform.Load().m_LocalToWorld;

        float3 NormalWorld = normalize(mul(LocalToWorld, float4(NormalLocal, 0.0f)).xyz);
        float3 LightDir = normalize(PushConst.m_LightDir.xyz);

        float Irradiance = max(dot(NormalWorld, -LightDir), 0.0f);
        float ShadowVisibility = PushConst.m_ShadowMask.SampleLevel(ESamplerType::SLinearClamp, AtlasUV, 0.0f);
        Irradiance *= ShadowVisibility;

        float3 Color = float3(1.0f) * Irradiance * 5.0f;

        PushConst.m_DirectLighting.Store(OverallOffset, float4(Color, 1.0f));
    }
}
}