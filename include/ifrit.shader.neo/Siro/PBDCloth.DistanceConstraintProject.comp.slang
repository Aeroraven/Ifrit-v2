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
        TRWStructuredBufferHandle<FPBDDistanceConstraint> m_DistanceConstraints;
        TRWStructuredBufferHandle<uint> m_FixedState;
        uint m_NumConstraints;
        float m_InvSolverIters;
        
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothDistanceConstraintProjectCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumConstraints) return;

        FPBDDistanceConstraint Constraint = PushConst.m_DistanceConstraints.Load(ThreadId.x);
        float3 PredPosA = PushConst.m_PredPositions.Load(Constraint.m_ParticleA).xyz;
        float3 PredPosB = PushConst.m_PredPositions.Load(Constraint.m_ParticleB).xyz;
        float RestLength = Constraint.m_RestLength;
        float Stiffness = Constraint.m_Stiffness;

        float3 Delta = PredPosA - PredPosB;
        float LengthDelta = length(Delta);
        float3 UnitDelta = Delta / (LengthDelta + 1e-8f); 

        float MassA = PushConst.m_InverseMass.Load(Constraint.m_ParticleA);
        float MassB = PushConst.m_InverseMass.Load(Constraint.m_ParticleB);

        uint FixedStateA = PushConst.m_FixedState.Load(Constraint.m_ParticleA);
        uint FixedStateB = PushConst.m_FixedState.Load(Constraint.m_ParticleB);

        if(FixedStateA != 0){
            MassA = 0.0;
        }

        if(FixedStateB != 0){
            MassB = 0.0;
        }

        float MassSum = MassA + MassB;
        float MassSumDeno = rcp(MassSum + 1e-9f);

        float MoveWeightA = -MassA * MassSumDeno;
        float MoveWeightB = MassB * MassSumDeno;

        float SpringComp = (LengthDelta - RestLength);
        float K = 1.0f - pow(1.0f - Stiffness, PushConst.m_InvSolverIters);

        float3 Correction = UnitDelta * SpringComp * K;
        float3 CorrectionA = Correction * MoveWeightA;
        float3 CorrectionB = Correction * MoveWeightB;

        if(FixedStateA == 0){
            PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleA, CorrectionA);
        }
        if(FixedStateB == 0){
            PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleB, CorrectionB);
        }
    }
}
}