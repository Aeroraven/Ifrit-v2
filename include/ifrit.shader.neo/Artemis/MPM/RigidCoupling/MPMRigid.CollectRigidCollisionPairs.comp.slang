#pragma ifrit.multi_compile IFSHADER_MPM_3D

#ifdef IFSHADER_MPM_3D
    #define IFSHADER_RIGID_DYNAMICS_3D
#endif

#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/RigidCoupling/MPMRigid.Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/RigidCoupling/MPMRigid.CollisionUtils.hlsli"

namespace IfritShader{
namespace Artemis{
namespace MPM{

    struct PushConst_T
    {
        uint m_NumRigids;
        TRWStructuredBufferHandle<FMPMRigidCouplingRigidContactPair> m_CollisionPairs;
        TAtomicRWStructuredBufferHandle<int> m_CollisionPairCounter;
        TStructuredBufferHandle<Rigid::FRigidColliderEntry> m_RigidColliders;
        TStructuredBufferHandle<Rigid::FRigidColliderDynamicsData> m_RigidDynamics;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;


    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MPMRigidCollectRigidCollisionPairsCS(uint3 ThreadId:SV_DispatchThreadID,
                                            uint3 GroupId:SV_GroupID,
                                            uint3 GroupThreadId:SV_GroupThreadID)
    {
        //TODO: broadphase->narrowphase
        int TotalPairs = PushConst.m_NumRigids * (PushConst.m_NumRigids - 1) / 2;
        if (ThreadId.x >= TotalPairs) return;
        int PairId = ThreadId.x;
        int RigidId1 = int(floor((-1.0 + sqrt(1.0 + 8.0 * PairId)) / 2.0));
        int RigidId2 = PairId - (RigidId1 * (RigidId1 + 1)) / 2;
        RigidId1 += 1;
        
        bool Valid = true;
        if (RigidId1 >= PushConst.m_NumRigids || RigidId2 >= PushConst.m_NumRigids || RigidId1 == RigidId2)
            Valid = false;

        FRigidRigidCollisionCheckResult Result;
        bool RigidCollided = false;
        int sCountPairsInWave = 0;
        if(Valid)
        {
            Rigid::FRigidColliderEntry Collider1 = PushConst.m_RigidColliders.Load(RigidId1);
            Rigid::FRigidColliderEntry Collider2 = PushConst.m_RigidColliders.Load(RigidId2);
            Rigid::FRigidColliderDynamicsData ColliderDynamics1 = PushConst.m_RigidDynamics.Load(Collider1.m_RuntimeId);
            Rigid::FRigidColliderDynamicsData ColliderDynamics2 = PushConst.m_RigidDynamics.Load(Collider2.m_RuntimeId);

            Result = DetectRigidRigidCollision(Collider1, ColliderDynamics1, Collider2, ColliderDynamics2);
            RigidCollided = Result.m_Collided;
        }

        
        sCountPairsInWave = WaveActiveCountBits(RigidCollided);
        int sCountPairsInWaveOffset = WavePrefixCountBits(RigidCollided);
        int sPairStartIndex = 0;
        if(WaveIsFirstLane())
        {
            sPairStartIndex = PushConst.m_CollisionPairCounter.AtomicAdd(0, sCountPairsInWave);
            PushConst.m_CollisionPairCounter.AtomicMax(1, DivRoundUp(sPairStartIndex + sCountPairsInWave,kMpmTGSizeX));
        }
        int OverallOffset = WaveReadLaneFirst(sPairStartIndex) + sCountPairsInWaveOffset;

        if(RigidCollided)
        {
            FMPMRigidCouplingRigidContactPair Pair;
            Pair.m_RigidId1 = RigidId1;
            Pair.m_RigidId2 = RigidId2;
            Pair.m_ContactNormal = SpatialVectorToFloat4(Result.m_ContactNormal);
            Pair.m_ContactPointRigidWrtCenter1 = SpatialVectorToFloat4(Result.m_ContactPointRigidWrtCenter1);
            Pair.m_ContactPointRigidWrtCenter2 = SpatialVectorToFloat4(Result.m_ContactPointRigidWrtCenter2);
            PushConst.m_CollisionPairs.Store(Pair, OverallOffset);
        }

    }

}}}
