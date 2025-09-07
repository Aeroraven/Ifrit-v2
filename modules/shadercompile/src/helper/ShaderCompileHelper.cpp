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
#include "ifrit/core/file/FileOps.h"

namespace Ifrit::ShaderCompile
{
    ShaderCompileOutput ShaderCompileHelper::CompileShaderFromSource(
        const ShaderCompileJob& job, EShaderIRFormat targetFormat)
    {
        auto                   sourceType = job.mSource.mFormat;
        IShaderSourceCompiler* compiler   = nullptr;
        if (sourceType == EShaderSourceFormat::GLSL && targetFormat == EShaderIRFormat::SpirV)
        {
            compiler = new GLSLProc::GlslSpirvTranslator();
        }
        else if (sourceType == EShaderSourceFormat::Slang && targetFormat == EShaderIRFormat::SpirV)
        {
            compiler = new SlangProc::SlangCompiler();
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader source format ");
            std::abort();
            return {};
        }
        compiler->SetCachePath(mCacheDir);
        compiler->SetIncludeBase(mIncludeBase);
        compiler->SetOptimization(mOptimization);
        auto output = compiler->Compile(job);
        delete compiler;
        return output;
    }

    ShaderCompileOutput ShaderCompileHelper::CompileShaderFromFile(const String& fileName, const String& entryPoint,
        const HashMap<String, String>& definitions, EShaderIRFormat targetFormat)
    {
        // get extension from fileName
        auto                extension         = fileName.substr(fileName.find_last_of('.') + 1);
        auto                remainingFileName = fileName.substr(0, fileName.find_last_of('.'));
        auto                stageName         = remainingFileName.substr(remainingFileName.find_last_of('.') + 1);

        EShaderSourceFormat sourceType;
        EShaderCompileStage stage;

        if (extension == "glsl")
        {
            sourceType = EShaderSourceFormat::GLSL;
        }
        else if (extension == "hlsl")
        {
            sourceType = EShaderSourceFormat::HLSL;
        }
        else if (extension == "slang")
        {
            sourceType = EShaderSourceFormat::Slang;
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader source format: {}", extension);
            std::abort();
            return {};
        }

        if (stageName == "vert" || stageName == "vs")
        {
            stage = EShaderCompileStage::VertexShader;
        }
        else if (stageName == "frag" || stageName == "fs")
        {
            stage = EShaderCompileStage::FragmentShader;
        }
        else if (stageName == "comp" || stageName == "cs")
        {
            stage = EShaderCompileStage::ComputeShader;
        }
        else if (stageName == "geom" || stageName == "gs")
        {
            stage = EShaderCompileStage::GeometryShader;
        }
        else if (stageName == "mesh" || stageName == "ms")
        {
            stage = EShaderCompileStage::MeshShader;
        }
        else if (stageName == "ampl" || stageName == "as")
        {
            stage = EShaderCompileStage::AmplificationShader;
        }
        else if (stageName == "task" || stageName == "ts")
        {
            stage = EShaderCompileStage::AmplificationShader;
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader compile stage: {}", stageName);
            std::abort();
            return {};
        }
        ShaderCompileJob job;
        job.mName           = fileName;
        job.mStage          = stage;
        job.mSource.mFormat = sourceType;
        job.mSource.mCode   = ReadTextFile(fileName);
        job.mEntryPoint     = entryPoint;
        job.mDefinitions    = definitions;

        return CompileShaderFromSource(job, EShaderIRFormat::SpirV);
    }

    IFRIT_APIDECL ShaderCompileHelper::~ShaderCompileHelper()
    {
        // Destructor implementation if needed
        // Currently, no dynamic memory is allocated in this class
    }

    IFRIT_APIDECL EShaderSourceFormat ShaderCompileHelper::GetEShaderSourceFormatFromFileName(const String& fileName)
    {
        // get extension from fileName
        auto extension = fileName.substr(fileName.find_last_of('.') + 1);

        if (extension == "glsl")
        {
            return EShaderSourceFormat::GLSL;
        }
        else if (extension == "hlsl")
        {
            return EShaderSourceFormat::HLSL;
        }
        else if (extension == "slang")
        {
            return EShaderSourceFormat::Slang;
        }
        else
        {
            IF_LOG_CRITICAL("ShaderCompiler", "Unsupported shader source format: {}", extension);
            std::abort();
            return EShaderSourceFormat::GLSL; // Default fallback
        }
    }

} // namespace Ifrit::ShaderCompile