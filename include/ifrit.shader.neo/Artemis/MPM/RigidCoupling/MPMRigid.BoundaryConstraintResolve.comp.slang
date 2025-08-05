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
        TStructuredBufferHandle<int> m_NumConstraints;
        TRWStructuredBufferHandle<FMPMRigidBoundaryContactPair> m_CollisionPairs;
        TStructuredBufferHandle<Rigid::FRigidColliderEntry> m_RigidColliders;
        FRigidColliderDynamicsHandle m_RigidDynamics;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MPMRigidBoundaryConstraintResolveCS(uint3 ThreadId:SV_DispatchThreadID,
                                             uint3 GroupId:SV_GroupID,
                                             uint3 GroupThreadId:SV_GroupThreadID)
    {
        int NumConstraints = PushConst.m_NumConstraints.Load(0);
        if (ThreadId.x >= NumConstraints) return;

        FMPMRigidBoundaryContactPair Contact = PushConst.m_CollisionPairs.Load(ThreadId.x);
        int RigidId = Contact.m_RigidId;

        Rigid::FRigidColliderEntry Collider = PushConst.m_RigidColliders.Load(RigidId);
        Rigid::FRigidColliderDynamicsData ColliderDynamics = PushConst.m_RigidDynamics.Load(Collider.m_RuntimeId);
        FScalar RigidMass = Collider.m_RigidMass;

        FSpatialVector ColliderCenterPositionPrev = ColliderDynamics.m_Position;
        FSpatialVector ColliderCenterDisplacement = ColliderDynamics.m_Displacement;
        FSpatialVector N = -ToSpatialVector(Contact.m_ContactNormal);

        FSpatialVector ColliderCenterPosition = ColliderCenterPositionPrev + ColliderCenterDisplacement;
        FSpatialVector P1 = ToSpatialVector(Contact.m_ContactPointWrtCenter) + ColliderCenterPosition;
        FSpatialVector P2 = ToSpatialVector(Contact.m_ContactPointBoundary);
        
        FScalar Dist = dot(P1 - P2, N);

        // dL = -Dist/(W1)
        // Imp = dL*N = -Dist/W1*N
        bool IsColliding = Dist > 0.0f;
        if(!IsColliding) return;

        FSpatialVector Impulse = -Dist * N;
        FSpatialVector Shift = Impulse;

        PushConst.m_RigidDynamics.AtomicAddDisplacement(Collider.m_RuntimeId, Shift);
    }

}}}
