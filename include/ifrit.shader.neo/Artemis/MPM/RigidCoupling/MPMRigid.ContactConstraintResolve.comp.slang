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

    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void MPMRigidContactConstraintResolveCS(uint3 ThreadId:SV_DispatchThreadID,
                                            uint3 GroupId:SV_GroupID,
                                            uint3 GroupThreadId:SV_GroupThreadID)
    {
        int NumConstraints = PushConst.m_NumConstraints.Load(0);
        if (ThreadId.x >= 1) return;

        for(int i=0;i<NumConstraints;i++)
        {
            FMPMRigidCouplingContactPair Contact = PushConst.m_CollisionPairs.Load(i);
            int RigidId = Contact.m_RigidId;
            int ParticleId = Contact.m_ParticleId;
    
            Rigid::FRigidColliderEntry Collider = PushConst.m_RigidColliders.Load(RigidId);
            Rigid::FRigidColliderDynamicsData ColliderDynamics = PushConst.m_RigidDynamics.Load(Collider.m_RuntimeId);
    
            FScalar ParticleMass = PushConst.m_ParticleMass.Load(ParticleId);
            FScalar RigidMass = Collider.m_RigidMass;
    
            FSpatialVector ParticlePositionPrev = PushConst.m_ParticleLocations.Load(ParticleId);
            FSpatialVector ParticleDisplacement = PushConst.m_ParticleDisplacements.Load(ParticleId);
    
            FSpatialVector ColliderContactPointLS = ToSpatialVector(Contact.m_ContactPointRigid);
            FSpatialVector ColliderContactPointWS = ConvertRigidPointToWorldSpace(ColliderContactPointLS, ColliderDynamics);
            
            FSpatialVector ParticlePosition = ParticlePositionPrev + ParticleDisplacement;
            FSpatialVector R2 = ColliderContactPointWS - (ColliderDynamics.m_Position + ColliderDynamics.m_Displacement);
            
            FSpatialVector N = ToSpatialVector(Contact.m_ContactNormal);
            FSpatialVector P1 = ParticlePosition;
            FSpatialVector P2 = ColliderContactPointWS;
    
            // TODO: Handle 3D rotation
            FScalar W2Extra = 0.0f;
    #ifndef IFSHADER_RIGID_DYNAMICS_3D
            W2Extra = Math::Cross2D(R2, N) * Math::Cross2D(R2, N) * rcp(Collider.m_Inertia2D);
    #endif
    
            // dx = (p1-p2)*n
            FScalar Dist = dot(P1 - P2, N);
            bool IsColliding = Dist > 0.0f;
            if (!IsColliding) return;
    
            FScalar InvM1 = rcp(ParticleMass);
            FScalar InvM2 = rcp(RigidMass) ;
            FScalar W1 = InvM1;
            FScalar W2 = InvM2 + W2Extra;
            FScalar DeltaLambda = -Dist/(W1 + W2);
    
            FSpatialVector Impulse = DeltaLambda * N;
    
            FSpatialVector Shift1 = InvM1 * Impulse;
            FSpatialVector Shift2 = -InvM2 * Impulse;
    
            Rigid::FAngularRotation RotationImp;
    
    #ifdef IFSHADER_RIGID_DYNAMICS_3D
            //TODO: Handle 3D rotation
    #else
            RotationImp = -rcp(Collider.m_Inertia2D) * Math::Cross2D(R2, Impulse);
            PushConst.m_RigidDynamics.AtomicAddRotation(Collider.m_RuntimeId, RotationImp);
    #endif
    
            PushConst.m_ParticleDisplacements.AtomicAdd(ParticleId, Shift1);
            PushConst.m_RigidDynamics.AtomicAddDisplacement(Collider.m_RuntimeId, Shift2);
        }
        
    }
}}}
