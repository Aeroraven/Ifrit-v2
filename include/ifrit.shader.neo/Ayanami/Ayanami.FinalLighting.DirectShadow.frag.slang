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
#include "ifrit.shader.neo/Lighting/CascadedShadowMapping.hlsli"


// This shader contains flaws, and will be integrated into
// the lighting shader in the future.
// Current implementation hinders the process of multi-lighting

namespace IfritShader {
namespace Ayanami {

    struct PushConstant_T {
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        uint m_TotalLights;
        TRWStructuredBufferHandle<Lighting::CascadedShadowMappingData> m_LightData;
        TTexture2DHandle<float> m_GBufferDepth;
        TTexture2DHandle<float4> m_GBufferNormal;
    };

    struct PixelInput
    {
        IFSHADER_LOCATION(0) float2 vTexCoord : TEXCOORD0;
    };

    struct PixelOutput
    {
        float ShadowMask : SV_Target0;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput FinalDirectShadowPS(PixelInput Input)
    {
        PerFramePerViewData PerFrame = PushConst.m_PerFrame.Load();

        float ClipDepth = PushConst.m_GBufferDepth.SampleLevel(ESamplerType::SNearestClamp, Input.vTexCoord, 0);
        float3 NdcPos = float3(Input.vTexCoord * 2.0f - 1.0f, ClipDepth);

        float4x4 ClipToWorld = PerFrame.m_ClipToWorld;
        float4x4 ClipToView = PerFrame.m_ClipToView;
        float4x4 ViewToWorld = PerFrame.m_ViewToWorld;

        float4 WorldPosH = mul(ClipToWorld, float4(NdcPos, 1.0f));
        float3 WorldPos = WorldPosH.xyz / WorldPosH.w;
        float4 ViewPosH = mul(ClipToView, float4(NdcPos, 1.0f));
        float3 ViewPos = ViewPosH.xyz / ViewPosH.w;

        float3 ViewNormal = PushConst.m_GBufferNormal.SampleLevel(ESamplerType::SNearestClamp, Input.vTexCoord, 0).xyz;
        ViewNormal = normalize(ViewNormal * 2.0f - 1.0f);
        float3 WorldNormal = mul(ViewToWorld, float4(ViewNormal, 0.0f)).xyz;
        WorldNormal = normalize(WorldNormal);

        float NormalOffset = 2e-2f;
        WorldPos += WorldNormal * NormalOffset;

        float2 ShadowVisibility = Lighting::GetCSMShadowVisibility(WorldPos, ViewPos, 0, PushConst.m_LightData);
        
        PixelOutput Output;
        Output.ShadowMask = ShadowVisibility.x;
        return Output;
    }

}
}