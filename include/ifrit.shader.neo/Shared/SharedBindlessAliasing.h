#pragma once

#ifndef __cplusplus
    #error "This file is only for C++ usage"
#endif

#ifdef __cplusplus
    #include "ifrit/core/base/IfritBasicAlias.h"
namespace Ifrit::Shader
{
    using BindlessIndexType                               = u32;
    template <typename T> using TRWStructuredBufferHandle = BindlessIndexType;
    template <typename T> using TRWTextureHandle          = BindlessIndexType;
    template <typename T> using TStructuredBufferHandle   = BindlessIndexType;
    template <typename T> using TTextureHandle            = BindlessIndexType;
} // namespace Ifrit::Shader

#endif