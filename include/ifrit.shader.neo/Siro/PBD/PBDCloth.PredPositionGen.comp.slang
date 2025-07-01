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

#include "ifrit.shader.neo/Siro/PBD/PBDCloth.Shared.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T{
        TRWStructuredBufferHandle<float4> m_PredPositions;
        TRWStructuredBufferHandle<float4> m_CurrentPositions;
        TRWStructuredBufferHandle<float4> m_Corrections;
        TRWStructuredBufferHandle<float> m_InverseMass;
        TRWStructuredBufferHandle<float4> m_Velocities;
        TRWStructuredBufferHandle<uint> m_FixedState;
        float m_DeltaTime;
        uint m_NumParticles;
        
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothPredPositionGenCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumParticles) return;

        float3 Velocity = PushConst.m_Velocities.Load(ThreadId.x).xyz;
        float InverseMass = PushConst.m_InverseMass.Load(ThreadId.x);
        float3 CurrentPosition = PushConst.m_CurrentPositions.Load(ThreadId.x).xyz;

        float3 PredPosition = CurrentPosition + Velocity * PushConst.m_DeltaTime;
        uint FixedState = PushConst.m_FixedState.Load(ThreadId.x);
        if (FixedState != 0) {
            PredPosition = CurrentPosition; 
        }
        PushConst.m_PredPositions.Store(float4(PredPosition, 0.0f), ThreadId.x);
        PushConst.m_Corrections.Store(float4(0.0f, 0.0f, 0.0f, 0.0f), ThreadId.x);
    }
}
}