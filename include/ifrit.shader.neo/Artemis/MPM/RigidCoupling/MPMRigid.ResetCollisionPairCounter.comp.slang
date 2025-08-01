#pragma ifrit.multi_compile IFSHADER_MPM_3D

#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        TAtomicRWStructuredBufferHandle<int> m_CollisionPairCounter;
        TAtomicRWStructuredBufferHandle<int> m_RigidCollisionPairCounter;
        TAtomicRWStructuredBufferHandle<int> m_BoundaryPairCounter;
        TAtomicRWStructuredBufferHandle<int> m_RigidParticleContactIndBuffer;
        TAtomicRWStructuredBufferHandle<int> m_RigidRigidContactIndBuffer;
        TAtomicRWStructuredBufferHandle<int> m_ParticleCounters;
        int m_NumRigids;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void MPMRigidResetCollisionPairCounterCS(uint3 ThreadId:SV_DispatchThreadID,
                                         uint3 GroupId:SV_GroupID,
                                         uint3 GroupThreadId:SV_GroupThreadID)
    {
        PushConst.m_CollisionPairCounter.Store(0, 0);
        PushConst.m_CollisionPairCounter.Store(0, 1);
        PushConst.m_CollisionPairCounter.Store(1, 2);
        PushConst.m_CollisionPairCounter.Store(1, 3);

        PushConst.m_RigidCollisionPairCounter.Store(0, 0);
        PushConst.m_RigidCollisionPairCounter.Store(0, 1);
        PushConst.m_RigidCollisionPairCounter.Store(1, 2);
        PushConst.m_RigidCollisionPairCounter.Store(1, 3);

        PushConst.m_BoundaryPairCounter.Store(0, 0);
        PushConst.m_BoundaryPairCounter.Store(0, 1);
        PushConst.m_BoundaryPairCounter.Store(1, 2);
        PushConst.m_BoundaryPairCounter.Store(1, 3);
        

        PushConst.m_RigidParticleContactIndBuffer.Store(DivRoundUp(PushConst.m_ParticleCounters.Load(0), kMpmTGSizeX),0);
        PushConst.m_RigidParticleContactIndBuffer.Store(PushConst.m_NumRigids,1);
        PushConst.m_RigidParticleContactIndBuffer.Store(1,2);
        PushConst.m_RigidParticleContactIndBuffer.Store(1,3);

        int TotalRigidRigidPairs = PushConst.m_NumRigids * (PushConst.m_NumRigids - 1) / 2;
        PushConst.m_RigidRigidContactIndBuffer.Store(DivRoundUp(TotalRigidRigidPairs, kMpmTGSizeX), 0);
        PushConst.m_RigidRigidContactIndBuffer.Store(1, 1);
        PushConst.m_RigidRigidContactIndBuffer.Store(1, 2);
        PushConst.m_RigidRigidContactIndBuffer.Store(1, 3);
    }

}}}
