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

namespace IfritShader{
namespace Ayanami{

    struct PixelInput
    {
        IFSHADER_LOCATION(0) float3 vMeshColor;
    };

    struct PixelOutput
    {
        IFSHADER_LOCATION(0) float4 outColor : SV_Target0;
    };

    struct PushConstant_T
    {
        float4 m_WorldBoundMin;
        float4 m_WorldBoundMax;
        TConstantBufferHandle<PerFramePerViewData> m_PerFrame;
        uint m_VoxelsPerWidth;
        uint m_ObjectGridId;
    };

    IFSHADER_PUSHCONST PushConstant_T PushConst;

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PixelOutput DebugVisObjectGridsPS(PixelInput input)
    {
        PixelOutput output;
        output.outColor = float4(input.vMeshColor, 1.0f);
        return output;
    }

}
}