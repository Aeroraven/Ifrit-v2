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

#include "spirv_reflect/spirv_reflect.c"

#include "ifrit/shadercompile/glslproc/GlslSpirvTranslator.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/typing/Util.h"
#include "sha1/sha1.hpp"
#include <shaderc/shaderc.hpp>
#include "spirv_reflect/spirv_reflect.h"

namespace Ifrit::ShaderCompile::GLSLProc
{
    class CustomShaderInclude : public shaderc::CompileOptions::IncluderInterface
    {
    private:
        Vec<String> m_includeDirs;

    public:
        CustomShaderInclude(const String& shaderDir) : m_shaderDir(shaderDir) {}

        shaderc_include_result* GetInclude(const char* requested_source, shaderc_include_type type,
            const char* requesting_source, size_t include_depth) override
        {
            String full_path = m_shaderDir + "/" + requested_source;
            m_includeDirs.push_back(full_path);

            std::ifstream file(full_path);
            if (!file.is_open())
            {
                return nullptr;
            }
            m_source = String((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            shaderc_include_result* result = new shaderc_include_result();
            result->content                = m_source.c_str();
            result->content_length         = m_source.size();
            result->source_name            = m_includeDirs.back().c_str();
            result->source_name_length     = m_includeDirs.back().size();
            result->user_data              = nullptr;
            return result;
        }

        void ReleaseInclude(shaderc_include_result* data) override { delete data; }

    private:
        String m_shaderDir;
        String m_source;
    };

    String PrecompileShaderFile(
        const String& source_name, shaderc_shader_kind kind, const String& source, const String& baseDir)
    {
        shaderc::Compiler       compiler;
        shaderc::CompileOptions options;
        options.SetIncluder(MakeOwner<CustomShaderInclude>(baseDir));
        options.SetGenerateDebugInfo();
        //  precompile
        shaderc::PreprocessedSourceCompilationResult precompiledModule =
            compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);
        if (precompiledModule.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            // std::cerr << source << std::endl;
            // std::cerr << precompiledModule.GetErrorMessage();
            IF_LOG_ERROR("GlslCompiler", "Failed to precompile shader: {}", source);
            IF_LOG_ERROR("GlslCompiler", "{}", precompiledModule.GetErrorMessage());
            std::abort();
        }

        return String(precompiledModule.cbegin(), precompiledModule.cend());
    }

    Vec<u32> CompileShaderFile(
        const String& source_name, shaderc_shader_kind kind, const String& source, const String& baseDir, bool optimize)
    {
        shaderc::Compiler       compiler;
        shaderc::CompileOptions options;
        options.SetIncluder(MakeOwner<CustomShaderInclude>(baseDir));
        options.SetGenerateDebugInfo();
        //  precompile
        shaderc::PreprocessedSourceCompilationResult precompiledModule =
            compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);
        if (precompiledModule.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            IF_LOG_ERROR("GlslCompiler", "Failed to precompile shader: {}", source_name);
            IF_LOG_ERROR("GlslCompiler", "{}", precompiledModule.GetErrorMessage());
            std::abort();
        }

        String preCode(precompiledModule.cbegin(), precompiledModule.cend());

        if (optimize)
            options.SetOptimizationLevel(shaderc_optimization_level_performance);
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(preCode, kind, source_name.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            IF_LOG_ERROR("GlslCompiler", "Failed to compile shader: {}", source_name);
            IF_LOG_ERROR("GlslCompiler", "{}", module.GetErrorMessage());

            std::abort();
            return Vec<u32>();
        }

        return { module.cbegin(), module.cend() };
    }

    shaderc_shader_kind GetShaderKind(ShaderCompileStage stage)
    {
        switch (stage)
        {
            case ShaderCompileStage::VertexShader:
                return shaderc_glsl_vertex_shader;
            case ShaderCompileStage::FragmentShader:
                return shaderc_glsl_fragment_shader;
            case ShaderCompileStage::ComputeShader:
                return shaderc_glsl_compute_shader;
            case ShaderCompileStage::GeometryShader:
                return shaderc_glsl_geometry_shader;
            case ShaderCompileStage::MeshShader:
                return shaderc_glsl_mesh_shader;
            case ShaderCompileStage::AmplificationShader:
                return shaderc_glsl_task_shader;
            default:
                IF_LOG_ASSERTION("GlslCompiler", false, "Unsupported shader stage for GlslSpirvTranslator");
                return shaderc_glsl_infer_from_source; // This should never be reached
        };
    }

    IFRIT_APIDECL ShaderCompileOutput GlslSpirvTranslator::Compile(const ShaderCompileJob& job)
    {

        ShaderCompileOutput output;

        // todo
        IF_LOG_ASSERTION("GlslCompiler", job.m_Source.m_Format == ShaderSourceFormat::GLSL,
            "GlslSpirvTranslator can only compile GLSL source code");
        IF_LOG_ASSERTION("GlslCompiler", job.m_EntryPoint == "main",
            "GlslSpirvTranslator only supports 'main' as the entry point for GLSL source code");

        // add permutations preprocessor definitions
        String rawCode = job.m_Source.m_Code;
        if (job.m_Stage == ShaderCompileStage::VertexShader)
        {
            rawCode = "#define IF_VERTEX_SHADER\n" + rawCode;
        }
        else if (job.m_Stage == ShaderCompileStage::FragmentShader)
        {
            rawCode = "#define IF_FRAGMENT_SHADER\n" + rawCode;
        }
        else if (job.m_Stage == ShaderCompileStage::ComputeShader)
        {
            rawCode = "#define IF_COMPUTE_SHADER\n" + rawCode;
        }
        else if (job.m_Stage == ShaderCompileStage::GeometryShader)
        {
            rawCode = "#define IF_GEOMETRY_SHADER\n" + rawCode;
        }
        else if (job.m_Stage == ShaderCompileStage::MeshShader)
        {
            rawCode = "#define IF_MESH_SHADER\n" + rawCode;
        }
        else if (job.m_Stage == ShaderCompileStage::AmplificationShader)
        {
            rawCode = "#define IF_AMPLIFICATION_SHADER\n" + rawCode;
        }

        if (!job.m_Definitions.empty())
        {
            for (const auto& def : job.m_Definitions)
            {
                rawCode = "#define " + def.first + " " + def.second + "\n" + rawCode;
            }
        }
        rawCode            = "#version 450\n" + rawCode;
        String precompiled = PrecompileShaderFile(job.m_Name, shaderc_glsl_vertex_shader, rawCode, m_IncludeBase);

        // calculate hash
        SHA1   sha1;
        sha1.update(precompiled);
        auto hash          = sha1.final();
        output.m_Signature = hash;

        Vec<u32> compiledCode;

        auto     cacheDir = m_CachePath;
        if (cacheDir.empty())
        {
            auto kind    = GetShaderKind(job.m_Stage);
            compiledCode = CompileShaderFile(
                job.m_Name, kind, rawCode, m_IncludeBase, m_Optimization == ShaderCompileOptimization::Performance);
            // iDebug("Code size: {}", compiledCode.size() * sizeof(u32));
        }
        else
        {
            String        cacheFile = cacheDir + "/ifritsc.spirv.shader." + hash + ".cache";
            // check if cache exists
            std::ifstream cache(cacheFile, std::ios::binary);
            if (cache.is_open())
            {
                cache.seekg(0, std::ios::end);
                size_t size = cache.tellg();
                cache.seekg(0, std::ios::beg);
                compiledCode.resize(size / sizeof(u32));
                cache.read(reinterpret_cast<char*>(compiledCode.data()), size);
                cache.close();
            }
            else
            {
                auto kind    = GetShaderKind(job.m_Stage);
                compiledCode = CompileShaderFile(
                    job.m_Name, kind, rawCode, m_IncludeBase, m_Optimization == ShaderCompileOptimization::Performance);
                std::ofstream cache(cacheFile, std::ios::binary);
                cache.write(reinterpret_cast<const char*>(compiledCode.data()), compiledCode.size() * sizeof(u32));
                cache.close();
            }
        }
        auto codeSize = compiledCode.size() * sizeof(u32);
        auto pCode    = compiledCode.data();

        // reflection data
        if (m_EnableReflection)
        {
            String                 cacheFile = cacheDir + "/ifritsc.sprivrefl." + hash + ".cache";
            SpvReflectShaderModule reflectionModule;

            if (cacheDir.empty())
            {
                spvReflectCreateShaderModule(compiledCode.size() * sizeof(u32), compiledCode.data(), &reflectionModule);
                u32 numDescSets = 0;
                spvReflectEnumerateDescriptorSets(&reflectionModule, &numDescSets, nullptr);
                // m_reflectSets.resize(numDescSets);
                // spvReflectEnumerateDescriptorSets(&reflectionModule, &numDescSets, m_reflectSets.data());
            }
            else
            {
                // check if cache exists
                std::ifstream cache(cacheFile, std::ios::binary);
                if (cache.is_open())
                {
                    cache.close();
                    // RecoverReflectionData();
                }
                else
                {
                    spvReflectCreateShaderModule(codeSize, pCode, &reflectionModule);
                    u32 numDescSets = 0;
                    spvReflectEnumerateDescriptorSets(&reflectionModule, &numDescSets, nullptr);
                    // m_reflectSets.resize(numDescSets);
                    // spvReflectEnumerateDescriptorSets(&reflectionModule, &numDescSets, m_reflectSets.data());
                    // m_reflectionCreated = true;
                    // CacheReflectionData();
                }
            }
        }
        output.m_IR.m_Format = ShaderIRFormat::SpirV;
        output.m_IR.m_Data.CopyFromRaw(pCode, SizeCast<u32>(codeSize));
        // iDebug("Data size: {}", output.m_IR.m_Data.GetSize());
        return output;
    }

    IFRIT_APIDECL GlslSpirvTranslator::~GlslSpirvTranslator()
    {
        // Cleanup if needed
        // Currently, no specific cleanup is required for this translator
    }

} // namespace Ifrit::ShaderCompile::GLSLProc