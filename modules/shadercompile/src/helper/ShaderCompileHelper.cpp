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

#include "ifrit/shadercompile/helper/ShaderCompileHelper.h"
#include "ifrit/shadercompile/glslproc/GlslSpirvTranslator.h"
#include "ifrit/shadercompile/slangproc/SlangCompiler.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::ShaderCompile
{
    ShaderCompileOutput ShaderCompileHelper::CompileShaderFromSource(
        const ShaderCompileJob& job, ShaderIRFormat targetFormat)
    {
        auto                   sourceType = job.m_Source.m_Format;
        IShaderSourceCompiler* compiler   = nullptr;
        if (sourceType == ShaderSourceFormat::GLSL && targetFormat == ShaderIRFormat::SpirV)
        {
            compiler = new GLSLProc::GlslSpirvTranslator();
        }
        else if (sourceType == ShaderSourceFormat::Slang && targetFormat == ShaderIRFormat::SpirV)
        {
            compiler = new SlangProc::SlangCompiler();
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader source format ");
            std::abort();
            return {};
        }
        compiler->SetCachePath(m_CacheDir);
        compiler->SetIncludeBase(m_IncludeBase);
        compiler->SetOptimization(m_Optimization);
        auto output = compiler->Compile(job);
        delete compiler;
        return output;
    }

    ShaderCompileOutput ShaderCompileHelper::CompileShaderFromFile(const String& fileName, const String& entryPoint,
        const HashMap<String, String>& definitions, ShaderIRFormat targetFormat)
    {
        // get extension from fileName
        auto               extension         = fileName.substr(fileName.find_last_of('.') + 1);
        auto               remainingFileName = fileName.substr(0, fileName.find_last_of('.'));
        auto               stageName         = remainingFileName.substr(remainingFileName.find_last_of('.') + 1);

        ShaderSourceFormat sourceType;
        ShaderCompileStage stage;

        if (extension == "glsl")
        {
            sourceType = ShaderSourceFormat::GLSL;
        }
        else if (extension == "hlsl")
        {
            sourceType = ShaderSourceFormat::HLSL;
        }
        else if (extension == "slang")
        {
            sourceType = ShaderSourceFormat::Slang;
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader source format: {}", extension);
            std::abort();
            return {};
        }

        if (stageName == "vert" || stageName == "vs")
        {
            stage = ShaderCompileStage::VertexShader;
        }
        else if (stageName == "frag" || stageName == "fs")
        {
            stage = ShaderCompileStage::FragmentShader;
        }
        else if (stageName == "comp" || stageName == "cs")
        {
            stage = ShaderCompileStage::ComputeShader;
        }
        else if (stageName == "geom" || stageName == "gs")
        {
            stage = ShaderCompileStage::GeometryShader;
        }
        else if (stageName == "mesh" || stageName == "ms")
        {
            stage = ShaderCompileStage::MeshShader;
        }
        else if (stageName == "ampl" || stageName == "as")
        {
            stage = ShaderCompileStage::AmplificationShader;
        }
        else if (stageName == "task" || stageName == "ts")
        {
            stage = ShaderCompileStage::AmplificationShader;
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader compile stage: {}", stageName);
            std::abort();
            return {};
        }
        ShaderCompileJob job;
        job.m_Name            = fileName;
        job.m_Stage           = stage;
        job.m_Source.m_Format = sourceType;
        job.m_Source.m_Code   = fileName; // In this case, the code is the file name
        job.m_EntryPoint      = entryPoint;
        job.m_Definitions     = definitions;

        return CompileShaderFromSource(job, ShaderIRFormat::SpirV);
    }

    IFRIT_APIDECL ShaderCompileHelper::~ShaderCompileHelper()
    {
        // Destructor implementation if needed
        // Currently, no dynamic memory is allocated in this class
    }

    IFRIT_APIDECL ShaderSourceFormat ShaderCompileHelper::GetShaderSourceFormatFromFileName(const String& fileName)
    {
        // get extension from fileName
        auto extension = fileName.substr(fileName.find_last_of('.') + 1);

        if (extension == "glsl")
        {
            return ShaderSourceFormat::GLSL;
        }
        else if (extension == "hlsl")
        {
            return ShaderSourceFormat::HLSL;
        }
        else if (extension == "slang")
        {
            return ShaderSourceFormat::Slang;
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader source format: {}", extension);
            std::abort();
            return ShaderSourceFormat::GLSL; // Default fallback
        }
    }

} // namespace Ifrit::ShaderCompile