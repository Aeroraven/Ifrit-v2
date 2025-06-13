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
#include "ifrit.shader.neo/Lighting/CascadedShadowMapping.hlsli"

#include "ifrit.shader.neo/PBR.hlsli"

namespace IfritShader {
namespace Ayanami {

    struct PixelInput {
        IFSHADER_LOCATION(0) float2 vTexCoord : TEXCOORD0;
    };

    struct PixelOutput {
        IFSHADER_LOCATION(0) float4 FinalColor : SV_Target0;
    };

    struct PushConstant_T {
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        TTexture2DHandle<float> m_ShadowOcclusion;
        TTexture2DHandle<float4> m_GBufferAlbedo;
        TTexture2DHandle<float4> m_GBufferNormal;
        TTexture2DHandle<float> m_GBufferDepth;
        uint m_TotalLights;
        TRWStructuredBufferHandle<Lighting::CascadedShadowMappingData> m_LightData;
    };


    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput FinalDirectLightingPS(PixelInput Input)
    {
        PerFramePerViewData PerFrame = PushConst.m_PerFrame.Load();

        float ClipDepth = PushConst.m_GBufferDepth.SampleLevel(ESamplerType::SNearestClamp, Input.vTexCoord, 0);
        if(ClipDepth <= 0.0f || ClipDepth >= 1.0f)
        {
            PixelOutput Output;
            Output.FinalColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
            return Output;
        }
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

        float3 Albedo = PushConst.m_GBufferAlbedo.SampleLevel(ESamplerType::SNearestClamp, Input.vTexCoord, 0).xyz;
        float ShadowOcclusion = PushConst.m_ShadowOcclusion.SampleLevel(ESamplerType::SNearestClamp, Input.vTexCoord, 0);

        float3 WorldNormal = mul(ViewToWorld, float4(ViewNormal, 0.0f)).xyz;
        WorldNormal = normalize(WorldNormal);

        float3 ViewDirVS = float4(normalize(ViewPos).xyz, 0.0f).xyz;
        float3 ViewDirWS = mul(ViewToWorld, float4(ViewDirVS, 0.0f)).xyz;
        ViewDirWS = normalize(ViewDirWS);

        float3 DirectLighting = float3(0.0f, 0.0f, 0.0f);
        // We only support one light for now (this is a temporary solution)
        
        float3 LightDirection = PushConst.m_LightData.Load(0).m_Views[0].Load().m_CameraFront.xyz;
        float3 Shading = PBRDirectLighting(
            LightDirection,
            WorldPos,
            WorldNormal,
            Albedo,
            ShadowOcclusion,
            -ViewDirWS
        );

        DirectLighting += Shading;
        
        PixelOutput Output;
        Output.FinalColor = float4(DirectLighting, 1.0f);

        return Output;
    }

} // namespace Ayanami
} // namespace IfritShader