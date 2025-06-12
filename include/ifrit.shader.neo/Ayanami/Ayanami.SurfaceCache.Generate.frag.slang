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
    struct PixelInput
    {
        IFSHADER_LOCATION(0) float2 TexCoord : TEXCOORD0;
        IFSHADER_LOCATION(1) float3 Normal : TEXCOORD1;
        IFSHADER_LOCATION(2) float4 Tangent : TEXCOORD2;
        IFSHADER_LOCATION(3) float4 WorldPos : TEXCOORD3;
    };

    struct PixelOutput
    {
        float4 Color : SV_Target0;
        float4 Normal : SV_Target1;
    };

    struct PushConstant_T
    {
        TTexture2DHandle<float4> m_Albedo;
        TTexture2DHandle<float2> m_Normal;
        TRWStructuredBufferHandle<float4> m_Object;
        TRWStructuredBufferHandle<float4> m_Card;
        TRWStructuredBufferHandle<float4> m_Vertex;
        TRWStructuredBufferHandle<float4> m_UV;
        TRWStructuredBufferHandle<float4> m_AllCardData;
        TRWStructuredBufferHandle<float4> m_TangentBuffer;
        TRWStructuredBufferHandle<float4> m_NormalBuffer;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;
    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput SurfaceCacheGeneratePS(PixelInput input)
    {
        float4 AlbedoColor = PushConst.m_Albedo.Sample(ESamplerType::SLinearClamp, input.TexCoord);
        float2 NormalColorRG = PushConst.m_Normal.Sample(ESamplerType::SLinearClamp, input.TexCoord);
        NormalColorRG = NormalColorRG * 2.0f - 1.0f; 
        float NormalB = sqrt(1.0f - dot(NormalColorRG, NormalColorRG));

        float3 Normal = float3(NormalColorRG, NormalB);
        float3 Tangent = normalize(input.Tangent.xyz);
        Tangent = normalize(Tangent - dot(Tangent,Normal) * Normal);

        float3 Bitangent = cross(Normal, Tangent);
        float3x3 TBN = float3x3(Tangent, Bitangent, Normal);

        float3 NormalLocal = normalize(mul(TBN, Normal));
        NormalLocal = NormalLocal * 0.5f + 0.5f; // Convert to [0,1] range

        PixelOutput output;
        output.Color = AlbedoColor;
        output.Normal = float4(Normal * 0.5f + 0.5f, 1.0f);
        
        return output;
    }
}