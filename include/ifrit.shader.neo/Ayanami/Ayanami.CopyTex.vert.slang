
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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

namespace IfritShader{

    struct VertexOutput
    {
        float4 Position : SV_Position;
        IFSHADER_LOCATION(0) float2 TexCoord : TEXCOORD0;
    };

    struct VertexInput
    {
        IFSHADER_LOCATION(0) float2 Position : POSITION;
    };

    IFSHADER_ENTRY(IFSHADER_VS_ENTRY)
    VertexOutput CopyTexVS(VertexInput input)
    {
        VertexOutput output;
        output.Position = float4(input.Position - float2(1.0f), 0.0f, 1.0f);
        output.TexCoord = input.Position * 0.5f;
        return output;
    }

}