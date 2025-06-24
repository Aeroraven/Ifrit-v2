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
        TRWStructuredBufferHandle<FPBDVolumeConstraint> m_VolumeConstraints;
        TRWStructuredBufferHandle<uint> m_FixedState;
        uint m_NumConstraints;
        float m_InvSolverIters;
        
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void SiroPBDClothVolumeConstraintProjectCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if (ThreadId.x >= PushConst.m_NumConstraints) return;

        FPBDVolumeConstraint Constraint = PushConst.m_VolumeConstraints.Load(ThreadId.x);
        
        float3 pA = PushConst.m_PredPositions.Load(Constraint.m_ParticleA).xyz;
        float3 pB = PushConst.m_PredPositions.Load(Constraint.m_ParticleB).xyz;
        float3 pC = PushConst.m_PredPositions.Load(Constraint.m_ParticleC).xyz;
        float3 pD = PushConst.m_PredPositions.Load(Constraint.m_ParticleD).xyz;

        float Volume = dot(cross(pB-pA,pC-pA), pD-pA) / 6.0f;

        float3 GradA = cross(pB-pC,pD-pC);
        float3 GradB = cross(pC-pA,pD-pA);
        float3 GradC = cross(pA-pB,pD-pB);
        float3 GradD = cross(pB-pA,pC-pA);

        float InvMassA = PushConst.m_InverseMass.Load(Constraint.m_ParticleA);
        float InvMassB = PushConst.m_InverseMass.Load(Constraint.m_ParticleB);
        float InvMassC = PushConst.m_InverseMass.Load(Constraint.m_ParticleC);
        float InvMassD = PushConst.m_InverseMass.Load(Constraint.m_ParticleD);

        uint FixedStateA = PushConst.m_FixedState.Load(Constraint.m_ParticleA);
        uint FixedStateB = PushConst.m_FixedState.Load(Constraint.m_ParticleB);
        uint FixedStateC = PushConst.m_FixedState.Load(Constraint.m_ParticleC);
        uint FixedStateD = PushConst.m_FixedState.Load(Constraint.m_ParticleD);

        if(FixedStateA>0) InvMassA = 0.0f;
        if(FixedStateB>0) InvMassB = 0.0f;
        if(FixedStateC>0) InvMassC = 0.0f;
        if(FixedStateD>0) InvMassD = 0.0f;

        float InvMassSum = InvMassA + InvMassB + InvMassC + InvMassD;
        if (InvMassSum <= 1e-9f) {
            return; 
        }
        float LambdaNumo = -(Volume-Constraint.m_RestVolume);
        float K = 1.0f - pow(1.0f - Constraint.m_Stiffness, PushConst.m_InvSolverIters);
        
        float Lambda = K * LambdaNumo / InvMassSum;

        float3 DeltaA = Lambda * GradA * InvMassA;
        float3 DeltaB = Lambda * GradB * InvMassB;
        float3 DeltaC = Lambda * GradC * InvMassC;
        float3 DeltaD = Lambda * GradD * InvMassD;

        PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleA, DeltaA);
        PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleB, DeltaB);
        PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleC, DeltaC);
        PushConst.m_Corrections.AddCorrection(Constraint.m_ParticleD, DeltaD);

    }
}
}