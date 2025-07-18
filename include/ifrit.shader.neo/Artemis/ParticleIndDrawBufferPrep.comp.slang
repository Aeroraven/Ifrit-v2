#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        TStructuredBufferHandle<int> m_ParticleCounter;
        TRWStructuredBufferHandle<int> m_IndirectDrawBuffer;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void ArtemisParticleIndirectDrawPrepCS(uint3 ThreadId:SV_DispatchThreadID,
                                           uint3 GroupId:SV_GroupID,
                                           uint3 GroupThreadId:SV_GroupThreadID)
    {
        uint ParticleCount = PushConst.m_ParticleCounter.Load(0);
        PushConst.m_IndirectDrawBuffer.Store(ParticleCount, 0);
        PushConst.m_IndirectDrawBuffer.Store(1, 1);
        PushConst.m_IndirectDrawBuffer.Store(0, 2);
        PushConst.m_IndirectDrawBuffer.Store(0, 3);
    }

}}}
