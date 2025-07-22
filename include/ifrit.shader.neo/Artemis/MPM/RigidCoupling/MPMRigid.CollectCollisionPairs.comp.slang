#pragma ifrit.multi_compile IFSHADER_MPM_3D

#ifdef IFSHADER_MPM_3D
#define IFSHADER_RIGID_DYNAMICS_3D
#endif

#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/RigidCoupling/MPMRigid.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        uint m_NumParticles;
        TStructuredBufferHandle<int> m_ParticleCounter;
        TRWStructuredBufferHandle<FMPMRigidCouplingContactPair> m_CollisionPairs;
        TAtomicRWStructuredBufferHandle<int> m_CollisionPairCounter;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MPMRigidCollectCollisionPairsCS(uint3 ThreadId:SV_DispatchThreadID,
                                         uint3 GroupId:SV_GroupID,
                                         uint3 GroupThreadId:SV_GroupThreadID)
    {
        // TODO: sparse datastructure or hierarchical culling for acceleration
        int RigidId = GroupId.y;
        uint NumParticles = PushConst.m_NumParticles;
        if(NumParticles == ~0u)
        {
            NumParticles = PushConst.m_ParticleCounter.Load(0);
        }

        int sCountPairsInWave = 0;
        bool ParticleCollided = false;
    }

}}}