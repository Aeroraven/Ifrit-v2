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
        float m_DeltaTime;
        float m_GravityNorm;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void MPMRigidSolveVelocityRigidRigidCollCS(uint3 ThreadId:SV_DispatchThreadID,
                                                uint3 GroupId:SV_GroupID,
                                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        // Handling restitution via XPBD velocity solver
        // Particle <-> Rigid Collision
        // dv = dot(n, -v_n+min(-e*v_{nt},0)) ; v_{nt} = last displacement @PB-MPM

        // TODO: Graph coloring
        
        int NumConstraints = PushConst.m_NumConstraints.Load(0);
        for(int i=0; i<NumConstraints; ++i)
        {
            FMPMRigidCouplingRigidContactPair Contact = PushConst.m_CollisionPairs.Load(i);
            int RigidId1 = Contact.m_RigidId1;
            int RigidId2 = Contact.m_RigidId2;

            Rigid::FRigidColliderEntry RigidCollider1 = PushConst.m_RigidColliders.Load(RigidId1);
            Rigid::FRigidColliderDynamicsData RigidDynamics1 = PushConst.m_RigidDynamics.Load(RigidCollider1.m_RuntimeId);
            Rigid::FRigidColliderEntry RigidCollider2 = PushConst.m_RigidColliders.Load(RigidId2);
            Rigid::FRigidColliderDynamicsData RigidDynamics2 = PushConst.m_RigidDynamics.Load(RigidCollider2.m_RuntimeId);
            
            FSpatialVector R1 = ConvertRigidPointWrtCenterDirToWS(ToSpatialVector(Contact.m_ContactPointRigidWrtCenter1), RigidDynamics1);
            FSpatialVector R2 = ConvertRigidPointWrtCenterDirToWS(ToSpatialVector(Contact.m_ContactPointRigidWrtCenter2), RigidDynamics2);

            Rigid::FAngularValue W1 = RigidDynamics1.m_AngularVelocity;
            Rigid::FAngularValue W1t = RigidDynamics1.m_AngularVelocityOld;
            Rigid::FAngularValue W2 = RigidDynamics2.m_AngularVelocity;
            Rigid::FAngularValue W2t = RigidDynamics2.m_AngularVelocityOld;
            

            FSpatialVector RotVelocity1 = FSpatialVector(0.0f);
            FSpatialVector RotVelocityT1 = FSpatialVector(0.0f);
            FSpatialVector RotVelocity2 = FSpatialVector(0.0f);
            FSpatialVector RotVelocityT2 = FSpatialVector(0.0f);
#ifndef IFSHADER_RIGID_DYNAMICS_3D
            RotVelocity1 = W1 * float2(-R1.y, R1.x);
            RotVelocityT1 = W1t * float2(-R1.y, R1.x);
            RotVelocity2 = W2 * float2(-R2.y, R2.x);
            RotVelocityT2 = W2t * float2(-R2.y, R2.x);
#endif
            FSpatialVector V2b = RigidDynamics2.m_Displacement;
            FSpatialVector V2 = V2b + RotVelocity2 * PushConst.m_DeltaTime;
            FSpatialVector V2bt = RigidDynamics2.m_DisplacementOld;
            FSpatialVector V2t = V2bt + RotVelocityT2 * PushConst.m_DeltaTime;

            FSpatialVector V1b = RigidDynamics1.m_Displacement;
            FSpatialVector V1 = V1b + RotVelocity1 * PushConst.m_DeltaTime;
            FSpatialVector V1bt = RigidDynamics1.m_DisplacementOld;
            FSpatialVector V1t = V1bt + RotVelocityT1 * PushConst.m_DeltaTime;

            FSpatialVector ContactNormal = ToSpatialVector(Contact.m_ContactNormal);
            FSpatialVector Vrel = V1 - V2;
            FSpatialVector VrelT = V1t - V2t;
            FScalar Vn = dot(ContactNormal, Vrel);
            FScalar Vnt = dot(ContactNormal, VrelT);

            if(abs(Vn) < 2*PushConst.m_GravityNorm * PushConst.m_DeltaTime * PushConst.m_DeltaTime)
            {
                //Vnt = 0.0f;
            }

            FSpatialVector Dv = ContactNormal * (-Vn + min(-Vnt, 0.0f));

            FScalar InvMass1 = rcp(RigidCollider1.m_RigidMass);
            FScalar InvMass2 = rcp(RigidCollider2.m_RigidMass);
            FScalar InvMass2Eff = 0.0f;
            FScalar InvMass1Eff = 0.0f;
#ifndef IFSHADER_RIGID_DYNAMICS_3D
            InvMass1Eff = InvMass1 + GetRotMass2D(RigidCollider1.m_Inertia2D, R1, ContactNormal);
            InvMass2Eff = InvMass2 + GetRotMass2D(RigidCollider2.m_Inertia2D, R2, ContactNormal);
#else
            //TODO: Handle 3D rotation
#endif

            FSpatialVector P = Dv / (InvMass1Eff + InvMass2Eff);
            FSpatialVector Pwt = P / PushConst.m_DeltaTime;

            FSpatialVector V1New = V1b + P * InvMass1;
            FSpatialVector V2New = V2b - P * InvMass2;

#ifndef IFSHADER_RIGID_DYNAMICS_3D
            FScalar Rot1Contrib = rcp(RigidCollider1.m_Inertia2D) * Math::Cross2D(R1, Pwt);
            FScalar Rot2Contrib = -rcp(RigidCollider2.m_Inertia2D) * Math::Cross2D(R2, Pwt);
            //RigidDynamics1.m_AngularVelocity = RigidDynamics1.m_AngularVelocity + Rot1Contrib;
            //RigidDynamics2.m_AngularVelocity = RigidDynamics2.m_AngularVelocity + Rot2Contrib;
#else
            //TODO: Handle 3D rotation
#endif

            RigidDynamics1.m_Displacement = V1New;
            PushConst.m_RigidDynamics.m_RigidDynamics.Store(RigidDynamics1, RigidCollider1.m_RuntimeId);
            RigidDynamics2.m_Displacement = V2New;
            PushConst.m_RigidDynamics.m_RigidDynamics.Store(RigidDynamics2, RigidCollider2.m_RuntimeId);
        }
    }
}}}
