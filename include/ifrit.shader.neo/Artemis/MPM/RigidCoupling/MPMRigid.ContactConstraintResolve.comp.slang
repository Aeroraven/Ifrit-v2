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
        if (ThreadId.x >= NumConstraints) return;

        FMPMRigidCouplingContactPair Contact = PushConst.m_CollisionPairs.Load(ThreadId.x);
        int RigidId = Contact.m_RigidId;
        int ParticleId = Contact.m_ParticleId;
        Rigid::FRigidColliderDynamicsData ColliderDynamics = PushConst.m_RigidDynamics.Load(RigidId);
        Rigid::FRigidColliderEntry Collider = PushConst.m_RigidColliders.Load(RigidId);

        FScalar ParticleMass = PushConst.m_ParticleMass.Load(ParticleId);
        FScalar RigidMass = PushConst.m_RigidColliders.Load(RigidId).m_RigidMass;

        FSpatialVector ParticlePositionPrev = PushConst.m_ParticleLocations.Load(ParticleId);
        FSpatialVector ColliderCenterPositionPrev = ColliderDynamics.m_Position;

        FSpatialVector ParticleDisplacement = PushConst.m_ParticleDisplacements.Load(ParticleId);
        FSpatialVector ColliderCenterDisplacement = ColliderDynamics.m_Displacement;

        FSpatialVector ParticlePosition = ParticlePositionPrev + ParticleDisplacement;
        FSpatialVector ColliderCenterPosition = ColliderCenterPositionPrev + ColliderCenterDisplacement;

        FSpatialVector N = ToSpatialVector(Contact.m_ContactNormal);
        FSpatialVector P1 = ParticlePosition;
        FSpatialVector P2 = ColliderCenterPosition + N * Collider.m_ColliderRadius;

        // dx = (p1-p2)*n
        FScalar Dist = -dot(P1 - P2, N);
        bool IsColliding = Dist > 0.0f;
        if (!IsColliding) return;

        FScalar W1 = rcp(ParticleMass);
        FScalar W2 = rcp(RigidMass);
        FScalar DeltaLambda = Dist/(W1 + W2);

        FSpatialVector Impulse = DeltaLambda * N;

        FSpatialVector Shift1 = W1 * Impulse;
        FSpatialVector Shift2 = -W2 * Impulse;

        FSpatialVector NewParticleDisplacement = ParticleDisplacement + Shift1;
        FSpatialVector NewColliderDisplacement = ColliderCenterDisplacement + Shift2;


        PushConst.m_ParticleDisplacements.AtomicAdd(ParticleId, Shift1);
        PushConst.m_RigidDynamics.AtomicAddDisplacement(RigidId, Shift2);
    }
}}}