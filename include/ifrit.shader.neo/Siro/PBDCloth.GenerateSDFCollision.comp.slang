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
#include "ifrit.shader.neo/Ayanami/Ayanami.SharedUtils.hlsli"
#include "ifrit.shader.neo/ObjectDescriptor.hlsli"

#include "ifrit.shader.neo/Math.RayTrace.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T{
        FPBDCollisionProcessIndirectArgs m_CollisionConstraintCounter;
        TRWStructuredBufferHandle<Ayanami::FMeshDFDetail> m_CollisionSDFs;
        TRWStructuredBufferHandle<FPerObjectData> m_ObjectData;
        TRWStructuredBufferHandle<FPBDCollsionConstraint> m_CollisionConstraints;
        TRWStructuredBufferHandle<float4> m_PredPositions;
        TRWStructuredBufferHandle<float4> m_Velocities;
        TRWStructuredBufferHandle<float4> m_Positions;
        uint m_NumParticles;
        uint m_NumSDFs;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothGenerateSDFCollisionCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumParticles) return;
        
        float3 TestSphereCenter = float3(0.5f, -1.5f, 0.5f);
        float TestSphereRadius = 1.0f;

        float3 ParticlePredPosition = PushConst.m_PredPositions.Load(ThreadId.x).xyz;
        float3 ParticleVelocity = PushConst.m_Velocities.Load(ThreadId.x).xyz;
        float3 ParticlePosition = PushConst.m_Positions.Load(ThreadId.x).xyz;

        bool bCollision = false;
        float3 CollisionPosition;
        float3 CollisionNormal;
        bCollision = Math::LineSphereIntersection(ParticlePosition,ParticlePredPosition,
            TestSphereCenter, TestSphereRadius, CollisionPosition);

        if (bCollision) {
            CollisionNormal = normalize(CollisionPosition - TestSphereCenter);

            uint PropId = PushConst.m_CollisionConstraintCounter.AddCounter(1);
            FPBDCollsionConstraint Constraint;
            Constraint.m_ParticleA = ThreadId.x;
            Constraint.m_CollisionNormal =  float4(CollisionNormal, 0.0f);
            Constraint.m_CollisionPos = float4(CollisionPosition, 0.0f);
            Constraint.m_CollisionVelocity = float4(ParticleVelocity, 0.0f);

            PushConst.m_CollisionConstraints.Store(Constraint, PropId);
        }
    }
}
}