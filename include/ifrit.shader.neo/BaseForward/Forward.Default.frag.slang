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

    struct PixelOutput
    {
        float4 Color : SV_Target0; // Albedo color
    };

    struct PushConstant_T
    {
        TRWStructuredBufferHandle<FPerObjectData> m_BatchedObjectData;
        TRWStructuredBufferHandle<PerFramePerViewData> m_PerFrameData;
        uint m_ObjectIndex; // Index of the current object in the batch
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput BaseForwardPS(VertexOutput VSOutput)
    {
        PixelOutput output;

        // Directly use normal to compute color
        output.Color = float4(VSOutput.vNormal * 0.5 + 0.5, 1.0); // Normal mapped color
        return output;
    }


}
}
