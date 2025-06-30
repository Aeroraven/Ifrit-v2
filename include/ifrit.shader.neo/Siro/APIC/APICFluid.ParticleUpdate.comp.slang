#include "ifrit.shader.neo/Siro/APIC/APICFluid.Utility.hlsli"

namespace IfritShader{
namespace Siro{

    struct PushConst_T
    {
        uint m_GridSize;
        uint m_NumParticles;
        float m_Timestep;
        TRWStructuredBufferHandle<float2> m_ParticlePositions;
        TRWStructuredBufferHandle<float2> m_ParticleVelocities;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSiroTGSizeX,1,1)
    void ApicParticleUpdateCS(uint3 ThreadId:SV_DispatchThreadID)
    {
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;
        
        float2 ParticlePosition = PushConst.m_ParticlePositions.Load(ThreadId.x);
        float2 ParticleVelocity = PushConst.m_ParticleVelocities.Load(ThreadId.x);

        float2 NewPosition = ParticlePosition + ParticleVelocity * PushConst.m_Timestep;

        float2 RegionMin = float2(0.0f, 0.0f);
        float2 RegionMax = float2((float)PushConst.m_GridSize - 1e-6f, (float)PushConst.m_GridSize - 1e-6f);

        NewPosition = clamp(NewPosition, RegionMin, RegionMax);
        PushConst.m_ParticlePositions.Store(NewPosition, ThreadId.x);

        float2 DefaultGravity = float2(0.0f, 9.81f);
        ParticleVelocity += DefaultGravity * PushConst.m_Timestep;

        PushConst.m_ParticleVelocities.Store(ParticleVelocity, ThreadId.x);
    }

}
}