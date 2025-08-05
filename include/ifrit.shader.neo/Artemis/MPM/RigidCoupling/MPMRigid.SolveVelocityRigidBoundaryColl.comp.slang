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
        float m_GravityNorm;
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

            FSpatialVector R1 = ConvertRigidPointWrtCenterDirToWS(ToSpatialVector(ContactPair.m_ContactPointWrtCenter), RigidDynamics);

            Rigid::FAngularValue W1 = RigidDynamics.m_AngularVelocity;
            Rigid::FAngularValue W1t = RigidDynamics.m_AngularVelocityOld;

            FSpatialVector RotVelocity = FSpatialVector(0.0f);
            FSpatialVector RotVelocityT = FSpatialVector(0.0f);
#ifndef IFSHADER_RIGID_DYNAMICS_3D
            RotVelocity = W1 * float2(-R1.y, R1.x);
            RotVelocityT = W1t * float2(-R1.y, R1.x);
#endif
            FSpatialVector Displacement = RigidDynamics.m_Displacement;// + RotVelocity * PushConst.m_DeltaTime;
            FSpatialVector V1 = Displacement + RotVelocity * PushConst.m_DeltaTime;
            FSpatialVector DisplacementT = RigidDynamics.m_DisplacementOld;// + RotVelocityT * PushConst.m_DeltaTime;  
            FSpatialVector V1t = DisplacementT + RotVelocityT * PushConst.m_DeltaTime;
            FSpatialVector ContactNormal = -ToSpatialVector(ContactPair.m_ContactNormal);
            FScalar Vn = dot(ContactNormal, V1);
            FScalar Vnt = dot(ContactNormal, V1t);
            if(abs(Vn) < 2 * PushConst.m_GravityNorm * PushConst.m_DeltaTime * PushConst.m_DeltaTime)
            {
                //Vnt = 0.0f;
            }

            FSpatialVector Dv = ContactNormal * (-Vn + min(-Vnt,0.0f));
            FScalar InvMass1 = rcp(RigidCollider.m_RigidMass);
            FScalar InvMass1Eff = 0.0f;
#ifndef IFSHADER_RIGID_DYNAMICS_3D
            InvMass1Eff = InvMass1 + GetRotMass2D(RigidCollider.m_Inertia2D, R1, ContactNormal);
#else
            //TODO: Handle 3D rotation
#endif
            FSpatialVector P = Dv / InvMass1Eff;
            FSpatialVector DvNew = P * InvMass1;
            FSpatialVector Pwt = DvNew / PushConst.m_DeltaTime;

            Rigid::FAngularValue W1New = RigidDynamics.m_AngularVelocity;
#ifndef IFSHADER_RIGID_DYNAMICS_3D
            W1New += rcp(RigidCollider.m_Inertia2D) * Math::Cross2D(R1,Pwt);
#else

#endif
            
            RigidDynamics.m_Displacement = Displacement + DvNew;
            RigidDynamics.m_AngularVelocity = W1New;
            PushConst.m_RigidDynamics.m_RigidDynamics.Store(RigidDynamics, RigidCollider.m_RuntimeId);
        }
    }
}}}
