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

namespace Ifrit::ShaderCompile
{
    enum class ShaderSourceFormat : u8
    {
        GLSL  = 0x01,
        HLSL  = 0x02,
        Slang = 0x03,
    };

    enum class ShaderIRFormat : u8
    {
        SpirV = 0x01,
        DXIL  = 0x02,
    };

    enum class ShaderCompileOptimization : u8
    {
        None        = 0x00,
        Size        = 0x01, // Optimize for size
        Performance = 0x02, // Optimize for speed
        Debug       = 0x03, // Debug mode, no optimizations
    };

    enum class ShaderCompileStage : u8
    {
        VertexShader        = 0x01,
        FragmentShader      = 0x02,
        ComputeShader       = 0x03,
        GeometryShader      = 0x04,
        MeshShader          = 0x10,
        AmplificationShader = 0x11,
    };

    struct ShaderSource
    {
        ShaderSourceFormat m_Format;
        String             m_Code;
    };

    struct ShaderIR
    {
        ShaderIRFormat m_Format;
        TSizedBuffer   m_Data;
    };

    struct ShaderCompileJob
    {
        String                  m_Name;
        ShaderSource            m_Source;
        String                  m_EntryPoint;
        HashMap<String, String> m_Definitions;
        ShaderCompileStage      m_Stage;
    };

    struct ShaderCompileOutput
    {
        ShaderIR m_IR;
        String   m_Signature;
    };

    class IFRIT_SHADERCOMPILE_API IShaderSourceCompiler
    {
    public:
        virtual void                SetCachePath(const String& path)                        = 0;
        virtual void                SetOptimization(ShaderCompileOptimization optimization) = 0;
        virtual void                SetIncludeBase(const String& base)                      = 0;
        virtual ShaderCompileOutput Compile(const ShaderCompileJob& job)                    = 0;
    };

    class IFRIT_SHADERCOMPILE_API ShaderCompilerBase : public IShaderSourceCompiler
    {
    protected:
        String                    m_CachePath;
        String                    m_IncludeBase;
        ShaderCompileOptimization m_Optimization = ShaderCompileOptimization::None;

    public:
        ShaderCompilerBase()          = default;
        virtual ~ShaderCompilerBase() = default;

        void                        SetCachePath(const String& path) override;
        void                        SetIncludeBase(const String& base);
        void                        SetOptimization(ShaderCompileOptimization optimization) override;

        virtual ShaderCompileOutput Compile(const ShaderCompileJob& job) override = 0;
    };

} // namespace Ifrit::ShaderCompile