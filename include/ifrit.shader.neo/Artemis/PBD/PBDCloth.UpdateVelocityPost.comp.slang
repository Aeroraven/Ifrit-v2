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

#include "ifrit.shader.neo/Artemis/PBD/PBDCloth.Shared.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T{
        float4 m_GravityFactor;
        TRWStructuredBufferHandle<float4> m_Positions;
        TRWStructuredBufferHandle<float4> m_Velocities;
        TRWStructuredBufferHandle<uint> m_FixedState;
        TRWStructuredBufferHandle<float4> m_PredPositions;
        float m_DeltaTime; 
        uint m_NumParticles;
        TRWStructuredBufferHandle<uint> m_CollisionCounters;
        
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ArtemisPBDClothUpdateVelocityPostCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        // clear counter
        PushConst.m_CollisionCounters.Store(0,0);
        PushConst.m_CollisionCounters.Store(0,1);
        
        if (ThreadId.x >= PushConst.m_NumParticles) return;
        uint FixedState = PushConst.m_FixedState.Load(ThreadId.x);
        if(FixedState != 0) return;
        
        float3 LastPosition = PushConst.m_Positions.Load(ThreadId.x).xyz;
        float3 PredictPosition = PushConst.m_PredPositions.Load(ThreadId.x).xyz;

        float3 EstVelocity = (PredictPosition - LastPosition) / PushConst.m_DeltaTime;
#ifndef IFSHADER_ARTEMIS_INDEPENDENT_UPDATE_VELOCITY_POST_KERNEL
        EstVelocity += PushConst.m_GravityFactor.xyz * PushConst.m_DeltaTime;
#endif

        PushConst.m_Velocities.Store(float4(EstVelocity, 0.0f), ThreadId.x);
        PushConst.m_Positions.Store(float4(PredictPosition, 1.0f), ThreadId.x);
    }
}
}