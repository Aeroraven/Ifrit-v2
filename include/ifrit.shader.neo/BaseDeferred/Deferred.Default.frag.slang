#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/ObjectDescriptor.hlsli"

namespace IfritShader {
namespace BaseDeferred{

    struct VertexOutput
    {
        float4 Position : SV_Position;
        IFSHADER_LOCATION(0) float2 vTexCoord : TEXCOORD0;
        IFSHADER_LOCATION(1) float3 vNormal : NORMAL;
        IFSHADER_LOCATION(2) float4 vTangent : TANGENT;
        IFSHADER_LOCATION(3) float4 vWorldPosition : TEXCOORD1;
    };

    struct PixelOutput
    {
        float4 Color : SV_Target0; // Albedo color
        float4 Normal : SV_Target1; // Normal vector
        float4 WorldPosition : SV_Target2; // World position
        float GelMask : SV_Target3; 
    };

    struct PushConstant_T
    {
        TRWStructuredBufferHandle<FPerObjectData> m_BatchedObjectData;
        TRWStructuredBufferHandle<PerFramePerViewData> m_PerFrameData;
        uint m_ObjectIndex; // Index of the current object in the batch
        uint m_MaskValue;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput BaseDeferredPS(VertexOutput VSOutput)
    {
        PixelOutput output;
        output.Color = float4(1.0f, 1.0f, 1.0f, 1.0f);
        output.Normal = float4(VSOutput.vNormal, 1.0f);
        output.WorldPosition = VSOutput.vWorldPosition;
        output.GelMask = PushConst.m_MaskValue > 0 ? 1.0f : 0.0f;
        return output;
    }


}
}
