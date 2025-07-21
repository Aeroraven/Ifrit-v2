#pragma once

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Bindless.hlsli"
#include "ifrit.shader.neo/Math.LinAlg.SpatialTransform.hlsli"

namespace IfritShader {
namespace Artemis{
namespace Rigid{

    IFSHADER_DEFINE_CONST_UINT32(kRigidTGSizeX, 128);

#ifndef __cplusplus

    struct FRigidColliderEntry
    {
        float4 m_Displacement;
        TRWStructuredBufferHandle<FInstanceLocalTransform> m_Transform;
        float m_ColliderRadius;
    };

#endif

}}}