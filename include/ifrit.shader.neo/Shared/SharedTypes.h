#pragma once
#include "ifrit.shader.neo/Shared/SharedCommon.h"

#ifdef __cplusplus
namespace Ifrit::Shader
{
#else
namespace IfritShader
{

#endif
    struct FInstanceLocalTransform
    {
        float4x4 m_LocalToWorld;
        float4x4 m_WorldToLocal;
        float4   m_MaxScale;
        float4   m_Position;
        float4   m_Rotation; // Euler!
    };

#ifdef __cplusplus
}
#else
}
#endif