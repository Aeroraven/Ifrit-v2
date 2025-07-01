#pragma ifrit.multi_compile IFSHADER_MPM_3D
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        int m_NumParticles;
        FScalar m_DeltaTime;
        FSpatialVectorHandle m_ParticleLocation;
        FSpatialVectorHandle m_ParticleVelocity;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MpmParticleAdvectionCS(uint3 ThreadId:SV_DispatchThreadID,
                               uint3 GroupId:SV_GroupID,
                               uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Step8. particle advection
        if(ThreadId.x >= PushConst.m_NumParticles)
            return;

        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FSpatialVector ParticleVelocity = PushConst.m_ParticleVelocity.Load(ThreadId.x);
        FSpatialVector NewLocation = ParticleLocation + PushConst.m_DeltaTime * ParticleVelocity;
        PushConst.m_ParticleLocation.Store(ThreadId.x, NewLocation);
    }

}}}