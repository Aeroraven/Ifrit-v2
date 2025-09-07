#include "ifrit.shader.neo/Artemis/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T
    {
        uint m_GridSize;
        uint m_NumParticles;
        TAtomicRWStructuredBufferHandle<float> m_GridVelocity;
        TAtomicRWStructuredBufferHandle<float> m_GridMass;
        TRWStructuredBufferHandle<float2> m_ParticlePositions;
        TRWStructuredBufferHandle<float> m_ParticleMass;
        TRWStructuredBufferHandle<float2> m_ParticleVelocities;
        TRWStructuredBufferHandle<float4> m_ParticleC; 
        TRWStructuredBufferHandle<int> m_GridState;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ApicParticleToGridCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        float4 ParticleCVector = PushConst.m_ParticleC.Load(ThreadId.x);
        
        float2x2 C;
        C[0][0] = ParticleCVector.x;
        C[0][1] = ParticleCVector.y;
        C[1][0] = ParticleCVector.z;
        C[1][1] = ParticleCVector.w;

        float2 ParticlePosition = PushConst.m_ParticlePositions.Load(ThreadId.x);
        float2 ParticleVelocity = PushConst.m_ParticleVelocities.Load(ThreadId.x);
        float ParticleMass = PushConst.m_ParticleMass.Load(ThreadId.x);

        int CenterGridX = int(ParticlePosition.x);
        int CenterGridY = int(ParticlePosition.y);

        var MomentumHandle = FApicGridVelocityHandle::Create(PushConst.m_GridSize, PushConst.m_GridVelocity);
        var MassHandle = FApicGridMassHandle::Create(PushConst.m_GridSize, PushConst.m_GridMass);
        
        [unroll]
        for(int dx = -2; dx <= 2; ++dx)
        {
            [unroll]
            for(int dy = -2; dy <= 2; ++dy)
            {
                int GridX = CenterGridX + dx;
                int GridY = CenterGridY + dy;
                int GridIndex = GetGridIndex(int2(GridX, GridY), PushConst.m_GridSize); 
                float2 GridCenter = float2(GridX, GridY) + 0.5f;
                if(!IsValidGridPos(int2(GridX, GridY), PushConst.m_GridSize))
                    continue;

                float2 GridCenterToParticle = GridCenter - ParticlePosition;
                float ContribWeightX = QuadraticInterpWeight(GridCenterToParticle.x);
                float ContribWeightY = QuadraticInterpWeight(GridCenterToParticle.y);

                float ContribWeight = ContribWeightX * ContribWeightY;

                if(ContribWeight <= 0.0f)
                    continue;

#ifndef IFSHADER_NO_APIC
                float2 ContribVelocity = ParticleVelocity + mul(C, GridCenterToParticle);
#else
                float2 ContribVelocity = ParticleVelocity;
#endif
                float2 WeightedMomentum = ContribWeight * ParticleMass * ContribVelocity;
                float WeightedMass = ContribWeight * ParticleMass; 

                MomentumHandle.AddVelocity(int2(GridX, GridY), WeightedMomentum);
                MassHandle.AddMass(int2(GridX, GridY), WeightedMass);
                PushConst.m_GridState.Store(1,GridIndex);
            }
        }
    }

}
}
