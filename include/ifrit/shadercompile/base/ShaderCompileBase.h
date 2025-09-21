/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/shadercompile/base/ShaderCompileApi.h"
#include "ifrit/core/algo/SizedBuffer.h"
#include "ifrit/core/reflection/Fwd.h"
#include "ifrit/core/base/containers/Maps.h"

namespace Ifrit::ShaderCompile
{
    enum class EShaderSourceFormat : u8
    {
        GLSL  = 0x01,
        HLSL  = 0x02,
        Slang = 0x03,
    };

    enum class EShaderIRFormat : u8
    {
        SpirV = 0x01,
        DXIL  = 0x02,
    };

    enum class EShaderCompileOptimization : u8
    {
        None        = 0x00,
        Size        = 0x01, // Optimize for size
        Performance = 0x02, // Optimize for speed
        Debug       = 0x03, // Debug mode, no optimizations
    };

    enum class EShaderCompileStage : u8
    {
        VertexShader        = 0x01,
        FragmentShader      = 0x02,
        ComputeShader       = 0x03,
        GeometryShader      = 0x04,
        MeshShader          = 0x10,
        AmplificationShader = 0x11,
    };

    enum class EShaderScalarType : u8
    {
        Unknown = 0x00,
        Int32   = 0x01,
        Uint32  = 0x02,
        Int64   = 0x03,
        Uint64  = 0x04,
        Float   = 0x05,
        Double  = 0x06,
        Bool    = 0x07,
    };

    enum class EShaderReflDescriptors : u8
    {
        Unknown = 0x00,

        ConstantBuffer     = 0x01,
        StructuredBuffer   = 0x02,
        RWStructuredBuffer = 0x03,
        Texture            = 0x04,
        RWTexture          = 0x05,
        SamplerState       = 0x06,

        BindlessHeap = 0x70,

        BindlessConstantBuffer     = 0x71,
        BindlessStructuredBuffer   = 0x72,
        BindlessRWStructuredBuffer = 0x73,
        BindlessTexture            = 0x74,
        BindlessRWTexture          = 0x75,
        BindlessSamplerState       = 0x76,

        DataInt32  = 0x40,
        DataUint32 = 0x41,
        DataInt64  = 0x42,
        DataUint64 = 0x43,
        DataFloat  = 0x44,
        DataDouble = 0x45,
        DataBool   = 0x46,
        DataVec2   = 0x47,
        DataVec3   = 0x48,
        DataVec4   = 0x49,
        DataVec2i  = 0x4A,
        DataVec3i  = 0x4B,
        DataVec4i  = 0x4C,
        DataVec2d  = 0x4D,
        DataVec3d  = 0x4E,
        DataVec4d  = 0x4F,
        DataVec2u  = 0x50,
        DataVec3u  = 0x51,
        DataVec4u  = 0x52,
        DataMat4f  = 0x53,
        DataMat2f  = 0x54,
    };

    struct ShaderSource
    {
        EShaderSourceFormat mFormat;
        String              mCode;
    };

    struct ShaderIR
    {
        EShaderIRFormat mFormat;
        SizedBuffer    mData;
    };

    struct IFRIT_SHADERCOMPILE_API ShaderBinding
    {
        String                 mName;
        EShaderReflDescriptors mType;
        u32                    mSet                = 0;
        u32                    mBinding            = 0;
        u32                    mArraySize          = 1; // 0 means unsized array
        u32                    mPushConstantOffset = 0; // only valid when type is PushConstant

        void                   DoSerialize(Reflection::Archive* archive) const;
        void                   DoDeserialize(Reflection::Archive* archive);
    };

    struct ShaderCompileJob
    {
        String                  mName;
        ShaderSource            mSource;
        String                  mEntryPoint;
        THashMap<String, String> mDefinitions;
        EShaderCompileStage     mStage;
        bool                    mRequestReflection = true;
    };

    struct IFRIT_SHADERCOMPILE_API ShaderReflectionData
    {
        Vec<ShaderBinding>   mBindings;
        THashMap<String, u32> mBindingNameToIndex;
        u32                  mPushConstantSize = 0;
        bool                 mValid            = false;

        void                 DoSerialize(Reflection::Archive* archive) const;
        void                 DoDeserialize(Reflection::Archive* archive);
    };

    struct ShaderCompileOutput
    {
        ShaderReflectionData mReflData;
        ShaderIR             mIR;
        String               mSignature;
    };

    class IFRIT_SHADERCOMPILE_API IShaderSourceCompiler
    {
    public:
        virtual void                SetCachePath(const String& path)                         = 0;
        virtual void                SetOptimization(EShaderCompileOptimization optimization) = 0;
        virtual void                SetIncludeBase(const String& base)                       = 0;
        virtual ShaderCompileOutput Compile(const ShaderCompileJob& job)                     = 0;
    };

    class IFRIT_SHADERCOMPILE_API ShaderCompilerBase : public IShaderSourceCompiler
    {
    protected:
        String                     mCachePath;
        String                     mIncludeBase;
        EShaderCompileOptimization mOptimization = EShaderCompileOptimization::None;

    public:
        ShaderCompilerBase()          = default;
        virtual ~ShaderCompilerBase() = default;

        void                        SetCachePath(const String& path) override;
        void                        SetIncludeBase(const String& base);
        void                        SetOptimization(EShaderCompileOptimization optimization) override;

        virtual ShaderCompileOutput Compile(const ShaderCompileJob& job) override = 0;
    };

} // namespace Ifrit::ShaderCompile