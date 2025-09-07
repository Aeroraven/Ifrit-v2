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

namespace IfritShader{
namespace Ayanami{

    struct VertexOutput
    {
        float4 Position : SV_Position;
        IFSHADER_LOCATION(0) float2 vTexCoord : TEXCOORD0;
        IFSHADER_LOCATION(1) float3 vNormal : NORMAL;
        IFSHADER_LOCATION(2) float4 vTangent : TANGENT;
        IFSHADER_LOCATION(3) float4 vWorldPosition : TEXCOORD1;
    };

    struct PushConstant_T
    {
        TTexture2DHandle<float4> m_Albedo;
        TTexture2DHandle<float2> m_Normal;
        TRWStructuredBufferHandle<float4> m_Object;
        uint m_CardId;
        TVertexDataHandle m_Vertex;
        TUVDataHandle m_UV;
        TRWStructuredBufferHandle<CardTransformData> m_AllCardData;
        TTangentDataHandle m_TangentBuffer;
        TNormalDataHandle m_NormalBuffer;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_VS_ENTRY)
    VertexOutput SurfaceCacheGenerateVS(uint VertexId : SV_VertexID)
    {
        float4 LocalPosition = PushConst.m_Vertex.Load(VertexId);
        float2 TexCoord = PushConst.m_UV.Load(VertexId);
        float3 Normal = PushConst.m_NormalBuffer.Load(VertexId).xyz;
        float4 Tangent = PushConst.m_TangentBuffer.Load(VertexId);

        float4x4 CardVP = PushConst.m_AllCardData.Load(PushConst.m_CardId).m_VP;
        float4 CardLocalPosition = mul(CardVP, LocalPosition);

        VertexOutput output;
        output.Position = CardLocalPosition;
        output.vTexCoord = TexCoord;
        output.vNormal = Normal;
        output.vTangent = Tangent;
        output.vWorldPosition = LocalPosition;

        return output;
    }



}
}