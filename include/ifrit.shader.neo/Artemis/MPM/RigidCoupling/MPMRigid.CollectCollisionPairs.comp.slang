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
        float4 ContactNormal = float4(0.0f, 0.0f, 0.0f, 0.0f);

        if(ParticleId < NumParticles)
        {
            Rigid::FRigidColliderEntry Collider = PushConst.m_RigidColliders.Load(RigidId);
            Rigid::FRigidColliderDynamicsData ColliderDynamics = PushConst.m_RigidDynamics.Load(RigidId);

            float ColliderRadius = Collider.m_ColliderRadius;
            FSpatialVector ColliderPositionPrev = ColliderDynamics.m_Position;
            FSpatialVector ParticlePositionPrev = PushConst.m_ParticlePositions.Load(ThreadId.x);

            FSpatialVector ParticleDisplacement = PushConst.m_ParticleDisplacements.Load(ThreadId.x);
            FSpatialVector ColliderDisplacement = ColliderDynamics.m_Displacement;

            FSpatialVector ParticlePosition = ParticlePositionPrev + ParticleDisplacement;
            FSpatialVector ColliderPosition = ColliderPositionPrev + ColliderDisplacement;

            FSpatialVector N = normalize(ParticlePosition - ColliderPosition);
            ContactNormal = SpatialVectorToFloat4(N);

            float Distance = length(ParticlePosition - ColliderPosition);
            ParticleCollided = Distance < ColliderRadius;

            if(ParticleCollided)
            {
                //PushConst.m_ParticleColor.Store(float4(1.0f, 0.0f, 0.0f, 1.0f), ParticleId);
            }
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
            Pair.m_Lambda = 0.0f;
            Pair.m_ContactNormal = ContactNormal;
            PushConst.m_CollisionPairs.Store(Pair, OverallOffset);
        }
    }

}}}
