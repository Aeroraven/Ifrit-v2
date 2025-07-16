#include "ifrit.shader.neo/Artemis/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T
    {
        uint m_GridSize;
        uint m_NumParticles;
        TRWStructuredBufferHandle<float2> m_ParticlePositions;
        TRWStructuredBufferHandle<float2> m_GridVelocity;
        TRWStructuredBufferHandle<float4> m_ParticleC; 
        TRWStructuredBufferHandle<float2> m_ParticleVelocity;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ApicGridToParticleCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        float2 ParticlePosition = PushConst.m_ParticlePositions.Load(ThreadId.x);
        int CenterGridX = int(ParticlePosition.x);
        int CenterGridY = int(ParticlePosition.y);

        float2x2 NewC = 0;
        float2 NewVelocity = float2(0.0f, 0.0f);
        float D = 4.0f; 

        for(int dx = -2; dx <= 2; ++dx)
        {
            for(int dy = -2; dy <= 2; ++dy)
            {
                int GridX = CenterGridX + dx;
                int GridY = CenterGridY + dy;
                int GridIndex = GetGridIndex(int2(GridX, GridY), PushConst.m_GridSize); 

                if(!IsValidGridPos(int2(GridX, GridY), PushConst.m_GridSize))
                    continue;

                float2 GridCenter = float2(GridX, GridY) + 0.5f;
                float2 GridCenterToParticle = GridCenter - ParticlePosition;
                float ContribWeightX = QuadraticInterpWeight(GridCenterToParticle.x);
                float ContribWeightY = QuadraticInterpWeight(GridCenterToParticle.y);

                float ContribWeight = ContribWeightX * ContribWeightY;
                float2 GridVelocity = PushConst.m_GridVelocity.Load(GridIndex);

                float2x2 B;
                B[0][0] = GridVelocity.x * GridCenterToParticle.x;
                B[0][1] = GridVelocity.x * GridCenterToParticle.y;
                B[1][0] = GridVelocity.y * GridCenterToParticle.x;
                B[1][1] = GridVelocity.y * GridCenterToParticle.y;
                

                NewC += ContribWeight * B * D;
                NewVelocity += ContribWeight * GridVelocity;
            }
        }

        float4 ParticleCVector;
        ParticleCVector.x = NewC[0][0];
        ParticleCVector.y = NewC[0][1];
        ParticleCVector.z = NewC[1][0];
        ParticleCVector.w = NewC[1][1];

        PushConst.m_ParticleC.Store(ParticleCVector, ThreadId.x);
        PushConst.m_ParticleVelocity.Store(NewVelocity, ThreadId.x);
    }

}
}
