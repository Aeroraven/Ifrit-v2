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
        TRWStructuredBufferHandle<FMPMRigidBoundaryContactPair> m_CollisionPairs;
        TStructuredBufferHandle<Rigid::FRigidColliderEntry> m_RigidColliders;
        FRigidColliderDynamicsHandle m_RigidDynamics;
        float m_DeltaTime;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void MPMRigidSolveVelocityRigidBoundaryCollCS(uint3 ThreadId:SV_DispatchThreadID,
                                                uint3 GroupId:SV_GroupID,
                                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Handling restitution via XPBD velocity solver
        // Boundary <-> Rigid Collision
        // dv = dot(n, -v_n+min(-e*v_{nt},0)) ; v_{nt} = last displacement @PB-MPM

        // TODO: Graph coloring
        
        int NumConstraints = PushConst.m_NumConstraints.Load(0);
        for(int i=0; i<NumConstraints; ++i)
        {
            FMPMRigidBoundaryContactPair ContactPair = PushConst.m_CollisionPairs.Load(i);
            Rigid::FRigidColliderEntry RigidCollider = PushConst.m_RigidColliders.Load(ContactPair.m_RigidId);
            Rigid::FRigidColliderDynamicsData RigidDynamics = PushConst.m_RigidDynamics.Load(RigidCollider.m_RuntimeId);

            FSpatialVector R1 = ToSpatialVector(ContactPair.m_ContactPoint) - RigidDynamics.m_Position;

            Rigid::FAngularValue W1 = RigidDynamics.m_AngularVelocity;
            Rigid::FAngularValue W1t = RigidDynamics.m_AngularVelocityOld;

            FSpatialVector RotVelocity = FSpatialVector(0.0f);
            FSpatialVector RotVelocityT = FSpatialVector(0.0f);
#ifndef IFSHADER_RIGID_DYNAMICS_3D
            RotVelocity = W1 * float2(-R1.y, R1.x);
            RotVelocityT = W1t * float2(-R1.y, R1.x);
#endif
            FSpatialVector Displacement = RigidDynamics.m_Displacement + RotVelocity * PushConst.m_DeltaTime;
            FSpatialVector DisplacementT = RigidDynamics.m_DisplacementOld + RotVelocityT * PushConst.m_DeltaTime;  
            FSpatialVector ContactNormal = -ToSpatialVector(ContactPair.m_ContactNormal);
            FSpatialVector Vn = dot(ContactNormal, Displacement);
            FSpatialVector Vnt = dot(ContactNormal, DisplacementT);

            FSpatialVector Dv = ContactNormal * (-Vn + min(-Vnt,0.0f));
            
            RigidDynamics.m_Displacement = Displacement + Dv;
            PushConst.m_RigidDynamics.m_RigidDynamics.Store(RigidDynamics, RigidCollider.m_RuntimeId);
        }
    }
}}}
