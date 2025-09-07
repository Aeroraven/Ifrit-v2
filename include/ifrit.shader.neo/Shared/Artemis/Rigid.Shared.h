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
        namespace Rigid
        {
#endif
    IFSHADER_SHARING_ENUMCLASS(ERigidColliderType, uint){ Sphere, Box };

    struct FRigidColliderEntry
    {
        float4                                             m_ColliderCuboidSize;
        ERigidColliderType                                 m_ColliderType;
        int                                                m_RuntimeId;
        TRWStructuredBufferHandle<FInstanceLocalTransform> m_Transform;
        float                                              m_ColliderRadius;
        float                                              m_RigidMass;
        float                                              m_Inertia2D;
        float                                              m_Padding1;
        float                                              m_Padding2;
    };

#ifdef __cplusplus
}
#else
        }
    }
}
#endif
