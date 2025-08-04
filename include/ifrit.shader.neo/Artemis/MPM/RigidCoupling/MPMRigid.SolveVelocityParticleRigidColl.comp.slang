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
        TRWStructuredBufferHandle<FMPMRigidCouplingContactPair> m_CollisionPairs;
        FSpatialVectorHandle m_ParticleLocations;
        TStructuredBufferHandle<Rigid::FRigidColliderEntry> m_RigidColliders;
        FRigidColliderDynamicsHandle m_RigidDynamics;

        FScalarHandle m_ParticleMass;
        FAtomicSpatialVectorHandle m_ParticleDisplacements;
        FAtomicSpatialVectorHandle m_ParticleDisplacementsOld;
        float m_DeltaTime;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(1, 1, 1)
    void MPMRigidSolveVelocityParticleRigidCollCS(uint3 ThreadId:SV_DispatchThreadID,
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
            FMPMRigidCouplingContactPair Contact = PushConst.m_CollisionPairs.Load(ThreadId.x);
            int RigidId = Contact.m_RigidId;
            int ParticleId = Contact.m_ParticleId;

            Rigid::FRigidColliderEntry RigidCollider = PushConst.m_RigidColliders.Load(RigidId);
            Rigid::FRigidColliderDynamicsData RigidDynamics = PushConst.m_RigidDynamics.Load(RigidCollider.m_RuntimeId);
            FSpatialVector R2 = PushConst.m_ParticleLocations.Load(ParticleId) - RigidDynamics.m_Position;
            Rigid::FAngularValue W2 = RigidDynamics.m_AngularVelocity;
            Rigid::FAngularValue W2t = RigidDynamics.m_AngularVelocityOld;

            FSpatialVector RotVelocity = FSpatialVector(0.0f);
            FSpatialVector RotVelocityT = FSpatialVector(0.0f);
#ifndef IFSHADER_RIGID_DYNAMICS_3D
            RotVelocity = W2 * float2(-R2.y, R2.x);
            RotVelocityT = W2t * float2(-R2.y, R2.x);
#endif
            FSpatialVector V2 = RigidDynamics.m_Displacement + RotVelocity * PushConst.m_DeltaTime;
            FSpatialVector V2t = RigidDynamics.m_DisplacementOld + RotVelocityT * PushConst.m_DeltaTime;

            FSpatialVector V1 = PushConst.m_ParticleDisplacements.Load(ParticleId);
            FSpatialVector V1t = PushConst.m_ParticleDisplacementsOld.Load(ParticleId);

            FSpatialVector ContactNormal = -ToSpatialVector(Contact.m_ContactNormal);
            FSpatialVector Vrel = V1 - V2;
            FSpatialVector VrelT = V1t - V2t;
            FSpatialVector Vn = dot(ContactNormal, Vrel);
            FSpatialVector Vnt = dot(ContactNormal, VrelT);

            FSpatialVector Dv = ContactNormal * (-Vn + min(-Vnt, 0.0f));

            FScalar InvMass1 = rcp(PushConst.m_ParticleMass.Load(ParticleId));
            FScalar InvMass2 = rcp(RigidCollider.m_RigidMass);

            FSpatialVector P = Dv / (InvMass1 + InvMass2);

            FSpatialVector V1New = V1 + P * InvMass1;
            FSpatialVector V2New = V2 - P * InvMass2;

            PushConst.m_ParticleDisplacements.Store(ParticleId, V1New);
            
            RigidDynamics.m_Displacement = V2New;
            PushConst.m_RigidDynamics.m_RigidDynamics.Store(RigidDynamics, RigidCollider.m_RuntimeId);
        }

    }
}}}
