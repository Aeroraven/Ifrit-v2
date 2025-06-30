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

#include "ifrit.shader.neo/Bindless.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T{
        TRWStructuredBufferHandle<float2> m_Positions;
        float m_ParticleRange;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    struct VSOutput
    {
        float4 Position: SV_Position;
        IFSHADER_POINTSIZE_DECORATE float PointSize: IFSHADER_POINTSIZE_SEMANTIC;
    };

    IFSHADER_ENTRY(IFSHADER_VS_ENTRY)
    VSOutput SiroParticleRenderVS(uint VertexId: SV_VertexID)
    {
        float2 Position = PushConst.m_Positions.Load(VertexId);
        float Range = PushConst.m_ParticleRange;

        VSOutput output;
        output.Position = float4(Position/ Range * 2.0 - 1.0, 0.5f, 1.0f) ;
        output.PointSize = 1.0f;

        return output;
    }
}
}