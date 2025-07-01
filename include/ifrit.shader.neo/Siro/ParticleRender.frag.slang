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
        float m_AspectRatio;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    struct PSOutput
    {
        IFSHADER_LOCATION(0) float4 Color: SV_Target0;
    };

    IFSHADER_ENTRY(IFSHADER_PS_ENTRY)
    PSOutput SiroParticleRenderPS()
    {
        PSOutput Output;
        Output.Color = float4(1.0f, 1.0f, 1.0f, 1.0f); 
        return Output;
    }
}
}