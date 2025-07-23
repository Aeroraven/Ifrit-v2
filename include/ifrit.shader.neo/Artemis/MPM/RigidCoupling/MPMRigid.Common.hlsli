#pragma once
#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader {
namespace Artemis {
namespace MPM{

    struct FMPMRigidCouplingContactPair
    {
        float4 m_ContactNormal;
        int m_ParticleId;
        int m_RigidId;
        float m_Lambda;
        int m_Pad;
    };

    struct FMPMRigidBoundaryContactPair
    {
        float4 m_ContactNormal;
        float4 m_ContactPoint;
        int m_RigidId;
    };


    struct FRigidColliderDynamicsHandle
    {
        TRWStructuredBufferHandle<Rigid::FRigidColliderDynamicsData> m_RigidDynamics;

        Rigid::FRigidColliderDynamicsData Load(int RigidId)
        {
            return m_RigidDynamics.Load(RigidId);
        }

        void AtomicAddDisplacement(int RigidId,FSpatialVector Displacement)
        {
            TAtomicRWStructuredBufferHandle<float> Casted;
            Casted.Index = m_RigidDynamics.Index;
            int Offset = RigidId * Rigid::kSizeofColliderDynamicsDataInF32;

            Casted.AtomicAdd(Offset + 0, Displacement.x);
            Casted.AtomicAdd(Offset + 1, Displacement.y);
#ifdef IFSHADER_RIGID_DYNAMICS_3D
            Casted.AtomicAdd(Offset + 2, Displacement.z);
#endif
        }

    }

}}}
