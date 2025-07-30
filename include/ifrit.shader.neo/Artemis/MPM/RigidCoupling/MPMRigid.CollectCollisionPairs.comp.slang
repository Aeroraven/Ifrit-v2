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
        uint m_NumParticles;
        TStructuredBufferHandle<int> m_ParticleCounter;
        TRWStructuredBufferHandle<FMPMRigidCouplingContactPair> m_CollisionPairs;
        TAtomicRWStructuredBufferHandle<int> m_CollisionPairCounter;
        FSpatialVectorHandle m_ParticlePositions;
        FSpatialVectorHandle m_ParticleDisplacements;
        TStructuredBufferHandle<Rigid::FRigidColliderEntry> m_RigidColliders;
        TStructuredBufferHandle<Rigid::FRigidColliderDynamicsData> m_RigidDynamics;

        // Debug Purpose
        TRWStructuredBufferHandle<float4> m_ParticleColor;
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
        int ParticleId = ThreadId.x;
        uint NumParticles = PushConst.m_NumParticles;
        if(NumParticles == ~0u)
        {
            NumParticles = PushConst.m_ParticleCounter.Load(0);
        }

        int sCountPairsInWave = 0;
        bool ParticleCollided = false;
        FParticleRigidCollisionCheckResult Result;

        if(ParticleId < NumParticles)
        {
            Rigid::FRigidColliderEntry Collider = PushConst.m_RigidColliders.Load(RigidId);
            Rigid::FRigidColliderDynamicsData ColliderDynamics = PushConst.m_RigidDynamics.Load(Collider.m_RuntimeId);

            float ColliderRadius = Collider.m_ColliderRadius;
            FSpatialVector ColliderPositionPrev = ColliderDynamics.m_Position;
            FSpatialVector ParticlePositionPrev = PushConst.m_ParticlePositions.Load(ThreadId.x);

            FSpatialVector ParticleDisplacement = PushConst.m_ParticleDisplacements.Load(ThreadId.x);
            FSpatialVector ColliderDisplacement = ColliderDynamics.m_Displacement;

            FSpatialVector ParticlePosition = ParticlePositionPrev + ParticleDisplacement;
            FSpatialVector ColliderPosition = ColliderPositionPrev + ColliderDisplacement;
            Result = DetectParticleRigidCollision(ParticlePosition, Collider, ColliderDynamics);
            ParticleCollided = Result.m_Collided;
        }

        sCountPairsInWave = WaveActiveCountBits(ParticleCollided);
        int sCountPairsInWaveOffset = WavePrefixCountBits(ParticleCollided);
        int sPairStartIndex = 0;
        if(WaveIsFirstLane())
        {
            sPairStartIndex = PushConst.m_CollisionPairCounter.AtomicAdd(0, sCountPairsInWave);
            PushConst.m_CollisionPairCounter.AtomicMax(1, DivRoundUp(sPairStartIndex + sCountPairsInWave,kMpmTGSizeX));
        }
        int OverallOffset = WaveReadLaneFirst(sPairStartIndex) + sCountPairsInWaveOffset;

        if(ParticleCollided)
        {
            FMPMRigidCouplingContactPair Pair;
            Pair.m_ParticleId = ParticleId;
            Pair.m_RigidId = RigidId;
            Pair.m_ContactNormal = SpatialVectorToFloat4(Result.m_ContactNormal);
            Pair.m_ContactPointRigid = SpatialVectorToFloat4(Result.m_CollisionLocRigid);
            PushConst.m_CollisionPairs.Store(Pair, OverallOffset);
        }
    }

}}}
