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

#include "ifrit.shader.neo/Siro/PBDCloth.Shared.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T{
        TRWStructuredBufferHandle<float4> m_ExternalForces;
        TRWStructuredBufferHandle<float> m_InverseMass;
        TRWStructuredBufferHandle<float4> m_Velocities;
        TRWStructuredBufferHandle<uint> m_FixedState;
        TRWStructuredBufferHandle<float4> m_PredPositions;
        float m_DeltaTime;
        float m_DampingFactor;
        uint m_NumParticles;
        
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothUpdateVelocityPreCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumParticles) return;

        float3 ExternalForce = PushConst.m_ExternalForces.Load(ThreadId.x).xyz;
        float InverseMass = PushConst.m_InverseMass.Load(ThreadId.x);
        float3 Velocity = PushConst.m_Velocities.Load(ThreadId.x).xyz;

        float3 NewVelocity = Velocity + ExternalForce * InverseMass * PushConst.m_DeltaTime;
#ifndef IFSHADER_SIRO_INDEPENDENT_DAMPING_KERNEL
        NewVelocity *= PushConst.m_DampingFactor;
#endif
        uint FixedState = PushConst.m_FixedState.Load(ThreadId.x);
        if (FixedState != 0) {
            NewVelocity = float3(0.0f, 0.0f, 0.0f); // If fixed, set velocity to zero
        }
        PushConst.m_Velocities.Store(float4(NewVelocity, 0.0f), ThreadId.x);
    }
}
}