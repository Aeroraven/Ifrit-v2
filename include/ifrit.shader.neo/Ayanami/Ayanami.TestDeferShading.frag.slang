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

    struct PushConstant_T {
        float4 m_LightDir;
        TTexture2DHandle<float4> m_Normal;
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        TTexture2DHandle<float> m_ShadowMap;
    };

    struct PixelInput
    {
        IFSHADER_LOCATION(0) float2 TexCoord : TEXCOORD0;
    };

    struct PixelOutput
    {
        IFSHADER_LOCATION(0) float4 Color : SV_Target0;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput TestDeferShadingPS(PixelInput Input)
    {
        PixelOutput Output = {};
        float4x4 ViewToWorld = PushConst.m_PerFrame.Load().m_ViewToWorld;
        float3 Normal = PushConst.m_Normal.SampleLevel(ESamplerType::SLinearClamp, Input.TexCoord, 0).xyz;
        if(dot(Normal,Normal) < 0.01f)
        {
            Output.Color = float4(0, 0, 0, 1);
            return Output;
        }
        Normal = Normal * 2.0f - 1.0f;
        Normal = normalize(mul(ViewToWorld, float4(Normal, 0)).xyz);
        float Shadow = PushConst.m_ShadowMap.SampleLevel(ESamplerType::SLinearClamp, Input.TexCoord, 0);
        float3 LightDir = normalize(PushConst.m_LightDir.xyz);
        float DotProduct = max(0.0f, dot(Normal, -LightDir)) * Shadow;

        Output.Color = float4(DotProduct, DotProduct, DotProduct, 1.0f) + 0.1f;
        return Output;
    }
}
}