#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/ObjectDescriptor.hlsli"

namespace IfritShader {
namespace BaseDeferred {

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
    VertexOutput BaseDeferredVS(uint VertexId : SV_VertexID)
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
