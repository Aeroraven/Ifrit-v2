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
        FPBDCollisionProcessIndirectArgs m_CollisionConstraintCounter;
        TRWStructuredBufferHandle<FPBDCollsionConstraint> m_CollisionConstraints;
        TRWStructuredBufferHandle<float4> m_PredPositions;
        FPBDCorrectionHandle m_CorrectionHandle;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothCollisionConstraintProjectCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_CollisionConstraintCounter.GetCounter()) return;

        FPBDCollsionConstraint Constraint = PushConst.m_CollisionConstraints.Load(ThreadId.x);  
        float3 Normal = Constraint.m_CollisionNormal.xyz;
        float3 CollPos = Constraint.m_CollisionPos.xyz;
        float3 PredPos = PushConst.m_PredPositions.Load(Constraint.m_ParticleA).xyz;
        float Cp = dot((PredPos - CollPos),Normal);
        float NsSq = dot(Normal, Normal);
        float InvNsSq = NsSq > 0.0f ? 1.0f / NsSq : 0.0f;

        float3 DeltaP = -Cp*InvNsSq * Normal;
        PushConst.m_CorrectionHandle.AddCorrection(Constraint.m_ParticleA,DeltaP);
    }
}
}