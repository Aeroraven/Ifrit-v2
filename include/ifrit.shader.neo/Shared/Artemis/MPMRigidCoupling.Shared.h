#pragma once
#include "ifrit.shader.neo/Shared/SharedCommon.h"
#include "ifrit.shader.neo/Shared/SharedTypes.h"

#ifdef __cplusplus
namespace Ifrit::Shader::Artemis
{
#else
namespace IfritShader
{
    namespace Artemis
    {
        namespace MPM
        {
#endif
    struct FMPMRigidCouplingContactPair
    {
        float4 m_ContactNormal;
        float4 m_ContactPointRigid; // in local space of rigid
        int    m_ParticleId;
        int    m_RigidId;
        int2   m_Pad;
    };
    struct FMPMRigidCouplingRigidContactPair
    {
        float4 m_ContactNormal;
        float4 m_ContactPointRigid1;
        float4 m_ContactPointRigid2;
        int    m_RigidId1;
        int    m_RigidId2;
        int2   m_Pad;
    };

    struct FMPMRigidBoundaryContactPair
    {
        float4 m_ContactNormal;
        float4 m_ContactPoint;
        int    m_RigidId;
    };

#ifdef __cplusplus
}
#else
        }
    }
}
#endif
