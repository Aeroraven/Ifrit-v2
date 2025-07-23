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
        uint m_NumRigids;
        TStructuredBufferHandle<Rigid::FRigidColliderEntry> m_RigidColliders;
        TStructuredBufferHandle<Rigid::FRigidColliderDynamicsData> m_RigidDynamics;
        FDenseGridStructureHandle m_MpmGrid;
        TRWStructuredBufferHandle<FMPMRigidBoundaryContactPair> m_CollisionPairs;
        TAtomicRWStructuredBufferHandle<int> m_CollisionPairCounter;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    void HandleBoundaryContact(int RigidId, FSpatialVector RigidCenter, FSpatialVector GridMinBound, 
        FSpatialVector GridMaxBound, FSpatialVector ContactNormal, bool IsColliding)
    {
        FSpatialVector RemainMask = FSpatialVector(1.0f) - abs(ContactNormal);
        FSpatialVector MinBoundContrib = max(FSpatialVector(0.0f),ContactNormal)* GridMinBound;
        FSpatialVector MaxBoundContrib = max(FSpatialVector(0.0f),-ContactNormal)* GridMaxBound;
        FSpatialVector RigidContrib = RemainMask * RigidCenter;
        FSpatialVector ContactPosition = MinBoundContrib + MaxBoundContrib + RigidContrib;

        int sCollidingPairsInWave = WaveActiveCountBits(IsColliding);
        int sCollidingPairsWaveOffset = WavePrefixCountBits(IsColliding);
        int sCollidingPairsGlobalStartOffset = 0;

        if(WaveIsFirstLane())
        {
            sCollidingPairsGlobalStartOffset = PushConst.m_CollisionPairCounter.AtomicAdd(0, sCollidingPairsInWave);
            int MaxPairs = sCollidingPairsGlobalStartOffset + sCollidingPairsInWave;
            PushConst.m_CollisionPairCounter.AtomicMax(1, DivRoundUp(MaxPairs, kMpmTGSizeX));
        }
        int OverallOffset = WaveReadLaneFirst(sCollidingPairsGlobalStartOffset) + sCollidingPairsWaveOffset;

        if(IsColliding)
        {
            FMPMRigidBoundaryContactPair ContactPair;
            ContactPair.m_RigidId = RigidId;
            ContactPair.m_ContactNormal = SpatialVectorToFloat4(ContactNormal);
            ContactPair.m_ContactPoint = SpatialVectorToFloat4(ContactPosition);
            PushConst.m_CollisionPairs.Store(ContactPair, OverallOffset);
        }
    }

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MPMRigidCollectBoundaryContactCS(uint3 ThreadId:SV_DispatchThreadID,
                                         uint3 GroupId:SV_GroupID,
                                         uint3 GroupThreadId:SV_GroupThreadID)
    {
        int RigidId = ThreadId.x;
        FDenseGridStructure Grid = PushConst.m_MpmGrid.Load();
        FSpatialVector GridMinBound = Grid.GetMinBoundWithBoundary();
        FSpatialVector GridMaxBound = Grid.GetMaxBoundWithBoundary();

        bool IsLeftPlaneIntersect = false;
        bool IsRightPlaneIntersect = false;
        bool IsTopPlaneIntersect = false;
        bool IsBottomPlaneIntersect = false;
        bool IsFrontPlaneIntersect = false;
        bool IsBackPlaneIntersect = false;

        FSpatialVector ColliderPosition = FSpatialVector(0.0f);
        FSpatialVector ColliderMinBound = FSpatialVector(0.0f);
        FSpatialVector ColliderMaxBound = FSpatialVector(0.0f);

        if(RigidId < PushConst.m_NumRigids)
        {
            Rigid::FRigidColliderEntry Collider = PushConst.m_RigidColliders.Load(RigidId);
            Rigid::FRigidColliderDynamicsData ColliderDynamics = PushConst.m_RigidDynamics.Load(RigidId);

            FSpatialVector ColliderPositionPrev = ColliderDynamics.m_Position;
            FSpatialVector ColliderDisplacement = ColliderDynamics.m_Displacement;
            FScalar ColliderRadius = Collider.m_ColliderRadius;

            ColliderPosition = ColliderPositionPrev + ColliderDisplacement;
            ColliderMinBound = ColliderPosition - FSpatialVector(ColliderRadius);
            ColliderMaxBound = ColliderPosition + FSpatialVector(ColliderRadius);

            IsLeftPlaneIntersect = ColliderMinBound.x <= GridMinBound.x;
            IsRightPlaneIntersect = ColliderMaxBound.x >= GridMaxBound.x;
            IsTopPlaneIntersect = ColliderMaxBound.y >= GridMaxBound.y;
            IsBottomPlaneIntersect = ColliderMinBound.y <= GridMinBound.y;
#ifdef IFSHADER_MPM_3D
            IsFrontPlaneIntersect = ColliderMaxBound.z >= GridMaxBound.z;
            IsBackPlaneIntersect = ColliderMinBound.z <= GridMinBound.z;
#endif
        }

        FSpatialVector LeftPlaneNormal = ToSpatialVector(float4(1.0f, 0.0f, 0.0f, 0.0f));
        FSpatialVector RightPlaneNormal = ToSpatialVector(float4(-1.0f, 0.0f, 0.0f, 0.0f));
        FSpatialVector TopPlaneNormal = ToSpatialVector(float4(0.0f, -1.0f, 0.0f, 0.0f));
        FSpatialVector BottomPlaneNormal = ToSpatialVector(float4(0.0f, 1.0f, 0.0f, 0.0f));
#ifdef IFSHADER_MPM_3D
        FSpatialVector FrontPlaneNormal = ToSpatialVector(float4(0.0f, 0.0f, 1.0f, 0.0f));
        FSpatialVector BackPlaneNormal = ToSpatialVector(float4(0.0f, 0.0f, -1.0f, 0.0f));
#endif

        HandleBoundaryContact(RigidId, ColliderPosition, GridMinBound, GridMaxBound, LeftPlaneNormal, IsLeftPlaneIntersect);
        HandleBoundaryContact(RigidId, ColliderPosition, GridMinBound, GridMaxBound, RightPlaneNormal, IsRightPlaneIntersect);
        HandleBoundaryContact(RigidId, ColliderPosition, GridMinBound, GridMaxBound, TopPlaneNormal, IsTopPlaneIntersect);
        HandleBoundaryContact(RigidId, ColliderPosition, GridMinBound, GridMaxBound, BottomPlaneNormal, IsBottomPlaneIntersect);
#ifdef IFSHADER_MPM_3D
        HandleBoundaryContact(RigidId, ColliderPosition, GridMinBound, GridMaxBound, FrontPlaneNormal, IsFrontPlaneIntersect);
        HandleBoundaryContact(RigidId, ColliderPosition, GridMinBound, GridMaxBound, BackPlaneNormal, IsBackPlaneIntersect);
#endif  
    }


}}}