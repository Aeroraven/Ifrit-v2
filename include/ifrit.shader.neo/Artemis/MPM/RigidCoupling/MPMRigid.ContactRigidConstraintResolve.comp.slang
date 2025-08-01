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
        TStructuredBufferHandle<int> m_NumConstraints;
        TRWStructuredBufferHandle<FMPMRigidCouplingRigidContactPair> m_CollisionPairs;
        TStructuredBufferHandle<Rigid::FRigidColliderEntry> m_RigidColliders;
        FRigidColliderDynamicsHandle m_RigidDynamics;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MPMRigidContactRigidConstraintResolveCS(uint3 ThreadId:SV_DispatchThreadID,
                                            uint3 GroupId:SV_GroupID,
                                            uint3 GroupThreadId:SV_GroupThreadID)
    {
        int NumConstraints = PushConst.m_NumConstraints.Load(0);
        if (ThreadId.x >= NumConstraints) return;

        FMPMRigidCouplingRigidContactPair Contact = PushConst.m_CollisionPairs.Load(ThreadId.x);
        int RigidId1 = Contact.m_RigidId1;
        int RigidId2 = Contact.m_RigidId2;

        Rigid::FRigidColliderEntry Collider1 = PushConst.m_RigidColliders.Load(RigidId1);
        Rigid::FRigidColliderDynamicsData ColliderDynamics1 = PushConst.m_RigidDynamics.Load(Collider1.m_RuntimeId);
        Rigid::FRigidColliderEntry Collider2 = PushConst.m_RigidColliders.Load(RigidId2);
        Rigid::FRigidColliderDynamicsData ColliderDynamics2 = PushConst.m_RigidDynamics.Load(Collider2.m_RuntimeId);

        FScalar RigidMass1 = Collider1.m_RigidMass;
        FScalar RigidMass2 = Collider2.m_RigidMass;

        FSpatialVector ColliderContactPointWS1 = ToSpatialVector(Contact.m_ContactPointRigid1);
        FSpatialVector ColliderContactPointWS2 = ToSpatialVector(Contact.m_ContactPointRigid2);
                
        FSpatialVector R1 = ColliderContactPointWS1 - (ColliderDynamics1.m_Position + ColliderDynamics1.m_Displacement);
        FSpatialVector R2 = ColliderContactPointWS2 - (ColliderDynamics2.m_Position + ColliderDynamics2.m_Displacement);
        
        FSpatialVector N = ToSpatialVector(Contact.m_ContactNormal);
        FSpatialVector P1 = ColliderContactPointWS1;
        FSpatialVector P2 = ColliderContactPointWS2;

        // TODO: Handle 3D rotation
        FScalar W1Extra = 0.0f;
        FScalar W2Extra = 0.0f;
#ifndef IFSHADER_RIGID_DYNAMICS_3D
        W1Extra = Math::Cross2D(R1, N) * Math::Cross2D(R1, N) * rcp(Collider1.m_Inertia2D);
        W2Extra = Math::Cross2D(R2, N) * Math::Cross2D(R2, N) * rcp(Collider2.m_Inertia2D);
#endif

        // dx = (p1-p2)*n
        FScalar Dist = -dot(P1 - P2, N);
        bool IsColliding = Dist > 0.0f;
        if (!IsColliding) return;

        FScalar W1 = rcp(RigidMass1) + W1Extra;
        FScalar W2 = rcp(RigidMass2) + W2Extra;
        FScalar DeltaLambda = Dist/(W1 + W2);

        FSpatialVector Impulse = DeltaLambda * N;

        FSpatialVector Shift1 = W1 * Impulse;
        FSpatialVector Shift2 = -W2 * Impulse;

        Rigid::FAngularRotation RotationImp1;
        Rigid::FAngularRotation RotationImp2;

#ifdef IFSHADER_RIGID_DYNAMICS_3D
        //TODO: Handle 3D rotation
#else
        RotationImp1 = -rcp(Collider1.m_Inertia2D) * Math::Cross2D(R1, Impulse);
        PushConst.m_RigidDynamics.AtomicAddRotation(Collider1.m_RuntimeId, RotationImp1);
        RotationImp2 = -rcp(Collider2.m_Inertia2D) * Math::Cross2D(R2, Impulse);
        PushConst.m_RigidDynamics.AtomicAddRotation(Collider2.m_RuntimeId, RotationImp2);
#endif
        PushConst.m_RigidDynamics.AtomicAddDisplacement(Collider1.m_RuntimeId, Shift1);
        PushConst.m_RigidDynamics.AtomicAddDisplacement(Collider2.m_RuntimeId, Shift2);
    }
}}}
