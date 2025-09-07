#include "ifrit.shader.neo/Artemis/APIC/APICFluid.Utility.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"

namespace IfritShader{
namespace Artemis{

    struct PushConst_T
    {
        uint m_GridSize;
        uint m_NumParticles;
        TRWStructuredBufferHandle<float2> m_ParticlePositions;
        TRWStructuredBufferHandle<float2> m_ParticleVelocities;
        TRWStructuredBufferHandle<float> m_ParticleMass;
        TRWStructuredBufferHandle<uint> m_ParticleIndex;
        TRWStructuredBufferHandle<float4> m_ParticleC;

    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kArtemisTGSizeX,1,1)
    void ApicParticleInitCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        float RandomSeedX = ThreadId.x * 114.514f;
        float RandomSeedY = ThreadId.x * 1919.0f;
        float RandomOffsetX = Random::Simplex2D(float2(RandomSeedX, RandomSeedY));

        float RandomSeedY2 = ThreadId.x * 1919.0f + 1.0f;
        float RandomOffsetY = Random::Simplex2D(float2(RandomSeedY2, RandomSeedX));

        float2 RandomOffset = float2(RandomOffsetX, RandomOffsetY);


        float2 Location = (RandomOffset - 0.5f) * PushConst.m_GridSize * 0.25f;
        Location += 0.5f * PushConst.m_GridSize;

        PushConst.m_ParticleC.Store(float4(0.0f, 0.0f, 0.0f, 0.0f), ThreadId.x);
        PushConst.m_ParticlePositions.Store(Location, ThreadId.x);
        PushConst.m_ParticleVelocities.Store(float2(0.0f, 0.0f), ThreadId.x);
        PushConst.m_ParticleMass.Store(1.0f, ThreadId.x);
        PushConst.m_ParticleIndex.Store(ThreadId.x, ThreadId.x);
    }

}
}
