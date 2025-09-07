#pragma once

#ifdef __cplusplus
    #include "ifrit/core/math/VectorDefs.h"
    #include "ifrit.shader.neo/Shared/SharedBindlessAliasing.h"
namespace Ifrit::Shader
{
    using float4   = Vector4f;
    using float2   = Vector2f;
    using uint     = u32;
    using int2     = Vector2i;
    using int4     = Vector4i;
    using uint2    = Vector2u;
    using uint4    = Vector4u;
    using float4x4 = Matrix4x4f;
    using float2x2 = Matrix2x2f;

    #define IFSHADER_SHARING_ENUMCLASS(name, underlying) enum class name : underlying

#else
    #include "ifrit.shader.neo/Bindless.hlsli"

    #define IFSHADER_SHARING_ENUMCLASS(name, underlying) enum name : underlying
namespace IfritShader
{
#endif

    // begin shared section

    // end shared section

#ifdef __cplusplus
}
#else
}
#endif