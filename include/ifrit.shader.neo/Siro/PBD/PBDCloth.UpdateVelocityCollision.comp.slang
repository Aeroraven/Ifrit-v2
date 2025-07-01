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
        TRWStructuredBufferHandle<float4> m_Positions;
        TRWStructuredBufferHandle<FPBDCollsionConstraint> m_CollisionConstraints;
        TRWStructuredBufferHandle<float4> m_Velocities;
        FPBDCollisionProcessIndirectArgs m_CollisionConstraintCounter;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothUpdateVelocityCollisionCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if(ThreadId.x >= PushConst.m_CollisionConstraintCounter.GetCounter()) return;

        FPBDCollsionConstraint Constraint = PushConst.m_CollisionConstraints.Load(ThreadId.x);
        float3 CollisionPos = Constraint.m_CollisionPos.xyz;
        float3 CollisionNormal = Constraint.m_CollisionNormal.xyz;
        float3 PredPosA = PushConst.m_Positions.Load(Constraint.m_ParticleA).xyz;

        float3 CollVelocity = PushConst.m_Velocities.Load(Constraint.m_ParticleA).xyz;

        float3 CollVelocityInNormal = dot(CollVelocity, CollisionNormal) * CollisionNormal;
        float3 CollVelocityInTangent = CollVelocity - CollVelocityInNormal;

        // Update velocity
        float3 NewVelocity = CollVelocityInTangent - CollVelocityInNormal * 0.3f;
        PushConst.m_Velocities.Store(float4(NewVelocity, 0.0f), Constraint.m_ParticleA);
    }
}
}