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
        TRWStructuredBufferHandle<float4> m_PredPositions;
        FPBDCorrectionHandle m_Corrections;
        TRWStructuredBufferHandle<float> m_InverseMass;
        TRWStructuredBufferHandle<FPBDBendingConstraint> m_BendingConstraints;
        TRWStructuredBufferHandle<uint> m_FixedState;
        uint m_NumConstraints;
        float m_InvSolverIters;
        
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothBendingConstraintProjectCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumConstraints) return;

        FPBDBendingConstraint Constraint = PushConst.m_BendingConstraints.Load(ThreadId.x);
        float InvMassA = PushConst.m_InverseMass.Load(Constraint.m_ParticleA);
        float InvMassB = PushConst.m_InverseMass.Load(Constraint.m_ParticleB);
        float InvMassC = PushConst.m_InverseMass.Load(Constraint.m_ParticleC);
        float InvMassD = PushConst.m_InverseMass.Load(Constraint.m_ParticleD);

        uint FixedStateA = PushConst.m_FixedState.Load(Constraint.m_ParticleA);
        uint FixedStateB = PushConst.m_FixedState.Load(Constraint.m_ParticleB);
        uint FixedStateC = PushConst.m_FixedState.Load(Constraint.m_ParticleC);
        uint FixedStateD = PushConst.m_FixedState.Load(Constraint.m_ParticleD);

        if(FixedStateA != 0){
            InvMassA = 0.0;
        }

        if(FixedStateB != 0){
            InvMassB = 0.0;
        }

        if(FixedStateC != 0){
            InvMassC = 0.0;
        }  

        if(FixedStateD != 0){
            InvMassD = 0.0;
        }

        float MassSum = InvMassA + InvMassB + InvMassC + InvMassD;

        float3 PosA = PushConst.m_PredPositions.Load(Constraint.m_ParticleA).xyz;
        float3 PosB = PushConst.m_PredPositions.Load(Constraint.m_ParticleB).xyz;
        float3 PosC = PushConst.m_PredPositions.Load(Constraint.m_ParticleC).xyz;
        float3 PosD = PushConst.m_PredPositions.Load(Constraint.m_ParticleD).xyz;

        float3 RelA = PosA - PosA;
        float3 RelB = PosB - PosA;
        float3 RelC = PosC - PosA;
        float3 RelD = PosD - PosA;

        float3 Cross23 = cross(RelB, RelC);
        float3 Cross24 = cross(RelB, RelD);
        float InvCross23Length = rsqrt(dot(Cross23, Cross23) + 1e-9f);
        float InvCross24Length = rsqrt(dot(Cross24, Cross24) + 1e-9f);

        float3 N1 = normalize(Cross23);
        float3 N2 = normalize(Cross24);

        float D = dot(N1, N2);

        float3 Q3 = (cross(RelB,N2) + cross(N1,RelB)*D) * InvCross23Length;
        float3 Q4 = (cross(RelB,N1) + cross(N2,RelB)*D) * InvCross24Length;
        float3 Q2A = - (cross(RelC,N2) + cross(N1,RelC)*D) * InvCross23Length;
        float3 Q2B = - (cross(RelD,N1) + cross(N2,RelD)*D) * InvCross24Length;
        float3 Q2 = (Q2A + Q2B);
        float3 Q1 = -Q2 - Q3 - Q4;

        float SharedNumo = sqrt(1.0f - D*D) * (acos(D)-Constraint.m_RestAngle);
        float SharedDeno = InvMassA*dot(Q1,Q1) + InvMassB*dot(Q2,Q2) + InvMassC*dot(Q3,Q3) + InvMassD*dot(Q4,Q4);
        float SharedFactor = -SharedNumo / (SharedDeno + 1e-9f);

        if(isnan(SharedFactor) || isinf(SharedFactor)){
            SharedFactor = 0.0f;
        }
        
        float K = 1.0f - pow(1.0f - Constraint.m_Stiffness, PushConst.m_InvSolverIters);
        SharedFactor *= K;

        float3 CorrectionA = Q1 * SharedFactor * InvMassA;
        float3 CorrectionB = Q2 * SharedFactor * InvMassB;
        float3 CorrectionC = Q3 * SharedFactor * InvMassC;
        float3 CorrectionD = Q4 * SharedFactor * InvMassD;

        if(FixedStateA == 0){
            PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleA, CorrectionA);
        }
        if(FixedStateB == 0){
            PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleB, CorrectionB);
        }
        if(FixedStateC == 0){
            PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleC, CorrectionC);
        }
        if(FixedStateD == 0){
            PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleD, CorrectionD);
        }

    }
}
}