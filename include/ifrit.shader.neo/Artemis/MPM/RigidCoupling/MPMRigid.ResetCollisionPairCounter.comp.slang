#pragma ifrit.multi_compile IFSHADER_MPM_3D

#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        TAtomicRWStructuredBufferHandle<int> m_CollisionPairCounter;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void MPMRigidResetCollisionPairCounterCS(uint3 ThreadId:SV_DispatchThreadID,
                                         uint3 GroupId:SV_GroupID,
                                         uint3 GroupThreadId:SV_GroupThreadID)
    {
        PushConst.m_CollisionPairCounter.Store(0, 0);
        PushConst.m_CollisionPairCounter.Store(1, 1);
        PushConst.m_CollisionPairCounter.Store(1, 2);
        PushConst.m_CollisionPairCounter.Store(1, 3);
    }

}}}