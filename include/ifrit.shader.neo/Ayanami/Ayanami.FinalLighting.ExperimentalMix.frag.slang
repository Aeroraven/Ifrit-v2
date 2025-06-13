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

namespace IfritShader {
namespace Ayanami{

    struct PixelInput
    {
        IFSHADER_LOCATION(0) float2 vTexCoord : TEXCOORD0;
    };

    struct PixelOutput
    {
        IFSHADER_LOCATION(0) float4 FinalColor : SV_Target0;
    };

    struct PushConstant_T
    {
        TTexture2DHandle<float4> m_DirectLighting;
        TTexture2DHandle<float4> m_IndirectLighting;
        TTexture2DHandle<float4> m_Albedo;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput FinalLightingMixPS(PixelInput Input)
    {
        PixelOutput Output;
        float4 DirectLighting = PushConst.m_DirectLighting.SampleLevel(ESamplerType::SLinearClamp, Input.vTexCoord, 0);
        float4 IndirectLighting = PushConst.m_IndirectLighting.SampleLevel(ESamplerType::SLinearClamp, Input.vTexCoord, 0);
        float4 Albedo = PushConst.m_Albedo.SampleLevel(ESamplerType::SLinearClamp, Input.vTexCoord, 0);

        float3 LambertBRDF = Albedo.rgb * kInvPI;
        float3 FinalColor = DirectLighting.xyz + IndirectLighting.xyz * LambertBRDF;

        Output.FinalColor = float4(FinalColor, 1.0f);
        return Output;
    }
}
}