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
#include "ifrit.shader.neo/ObjectDescriptor.hlsli"

namespace IfritShader {
namespace BaseForward{

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
        TRWStructuredBufferHandle<FPerObjectData> m_BatchedObjectData;
        TStructuredBufferHandle<PerFramePerViewData> m_PerFrameData;
        uint m_ObjectIndex; // Index of the current object in the batch
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_VS_ENTRY)
    VertexOutput BaseForwardVS(uint VertexId : SV_VertexID)
    {
        FPerObjectData ObjectData = PushConst.m_BatchedObjectData.Load(PushConst.m_ObjectIndex);
        FMeshData MeshData = ObjectData.m_MeshData.Load(0);
        float4x4 ObjectLocal = ObjectData.m_Transform.Load(0).m_LocalToWorld;
        float4x4 WorldToClip = PushConst.m_PerFrameData.Load(0).m_WorldToClip;
        float4x4 ObjectToClip = mul(WorldToClip, ObjectLocal);


        float4 LocalPosition = MeshData.m_VertexBuffer.Load(VertexId);
        float2 TexCoord = MeshData.m_UVBuffer.Load(VertexId);
        float3 Normal = MeshData.m_NormalBuffer.Load(VertexId).xyz;
        float4 Tangent = MeshData.m_TangentBuffer.Load(VertexId);

        float4 FinalPos = mul(ObjectToClip, LocalPosition);

        VertexOutput output;
        output.Position = FinalPos;
        output.vTexCoord = TexCoord;
        output.vNormal = Normal;
        output.vTangent = Tangent;
        output.vWorldPosition = LocalPosition;

        return output;
    }


}
}
