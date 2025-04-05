
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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

#include "ifrit/core/typing/Util.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Shader.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include "sha1/sha1.hpp"
#include <fstream>
#include <iostream>
#include <shaderc/shaderc.hpp>

namespace Ifrit::Graphics::VulkanGraphics
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

    String precompileShaderFile(const String& source_name, shaderc_shader_kind kind, const String& source)
    {
        shaderc::Compiler       compiler;
        shaderc::CompileOptions options;
        options.SetIncluder(std::make_unique<CustomShaderInclude>(IFRIT_VKGRAPHICS_SHARED_SHADER_PATH));
        options.SetGenerateDebugInfo();
        //  precompile
        shaderc::PreprocessedSourceCompilationResult precompiledModule =
            compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);
        if (precompiledModule.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            std::cerr << source << std::endl;
            std::cerr << precompiledModule.GetErrorMessage();
            std::abort();
        }

        return String(precompiledModule.cbegin(), precompiledModule.cend());
    }

    Vec<u32> compileShaderFile(
        const String& source_name, shaderc_shader_kind kind, const String& source, bool optimize = true)
    {
        shaderc::Compiler       compiler;
        shaderc::CompileOptions options;
        options.SetIncluder(std::make_unique<CustomShaderInclude>(IFRIT_VKGRAPHICS_SHARED_SHADER_PATH));
        options.SetGenerateDebugInfo();
        //  precompile
        shaderc::PreprocessedSourceCompilationResult precompiledModule =
            compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);
        if (precompiledModule.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            std::cerr << source << std::endl;
            std::cerr << precompiledModule.GetErrorMessage();
            std::abort();
        }

        String preCode(precompiledModule.cbegin(), precompiledModule.cend());

        if (optimize)
            options.SetOptimizationLevel(shaderc_optimization_level_performance);
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

        shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(preCode, kind, source_name.c_str(), options);

        if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        {
            std::cerr << "During compilation of:" << source_name << std::endl;
            std::cerr << module.GetErrorMessage();

            std::abort();
            return Vec<u32>();
        }

        return { module.cbegin(), module.cend() };
    }

    IFRIT_APIDECL ShaderModule::ShaderModule(EngineContext* ctx, const ShaderModuleCI& ci)
    {
        m_context = ctx;
        VkShaderModuleCreateInfo moduleCI{};
        shaderc_shader_kind      kind;
        if (ci.stage == Rhi::RhiShaderStage::Vertex)
        {
            m_stageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
            kind            = shaderc_vertex_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::Fragment)
        {
            m_stageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            kind            = shaderc_fragment_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::Compute)
        {
            m_stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            kind            = shaderc_compute_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::Mesh)
        {
            m_stageCI.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
            kind            = shaderc_mesh_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::Task)
        {
            m_stageCI.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
            kind            = shaderc_task_shader;
        }
        // Raytracing
        else if (ci.stage == Rhi::RhiShaderStage::RTRayGen)
        {
            m_stageCI.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            kind            = shaderc_raygen_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::RTClosestHit)
        {
            m_stageCI.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            kind            = shaderc_closesthit_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::RTMiss)
        {
            m_stageCI.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
            kind            = shaderc_miss_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::RTAnyHit)
        {
            m_stageCI.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            kind            = shaderc_anyhit_shader;
        }
        else if (ci.stage == Rhi::RhiShaderStage::RTIntersection)
        {
            m_stageCI.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
            kind            = shaderc_intersection_shader;
        }

        Vec<u32> compiledCode;
        if (ci.sourceType == Rhi::RhiShaderSourceType::GLSLCode)
        {

            auto   cacheDir = ctx->GetCacheDir();

            // Shader cache is a temporary solution,
            // PSO cache should be used in the future
            SHA1   sha1;
            String rawCode(ci.code.begin(), ci.code.end());
            String precompiled;
            precompiled = precompileShaderFile(ci.fileName, static_cast<shaderc_shader_kind>(kind), rawCode);
            sha1.update(precompiled);
            auto hash   = sha1.final();
            m_signature = hash;

            if (cacheDir.empty())
            {
                compiledCode      = compileShaderFile(ci.fileName, static_cast<shaderc_shader_kind>(kind), rawCode);
                moduleCI.codeSize = compiledCode.size() * sizeof(u32);
                moduleCI.pCode    = compiledCode.data();
            }
            else
            {
                String        cacheFile = cacheDir + "/vkgraphics.shader." + hash + ".cache";
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
                    compiledCode = compileShaderFile(ci.fileName, static_cast<shaderc_shader_kind>(kind), rawCode);
                    std::ofstream cache(cacheFile, std::ios::binary);
                    cache.write(reinterpret_cast<const char*>(compiledCode.data()), compiledCode.size() * sizeof(u32));
                    cache.close();
                }
            }
            moduleCI.codeSize = compiledCode.size() * sizeof(u32);
            moduleCI.pCode    = compiledCode.data();
        }
        else
        {
            moduleCI.codeSize = ci.code.size();
            moduleCI.pCode    = reinterpret_cast<const u32*>(ci.code.data());
        }
        VkDevice device = m_context->GetDevice();
        moduleCI.sType  = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

        vkrVulkanAssert(vkCreateShaderModule(device, &moduleCI, nullptr, &m_module), "Failed to create shader module");
        m_stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

        m_ci             = ci;
        m_stageCI.module = m_module;
        m_stageCI.pName  = m_ci.entryPoint.c_str();
        m_stageCI.flags  = 0;
        m_stageCI.pNext  = nullptr;
        // for spirv reflect
        auto   cacheDir  = ctx->GetCacheDir();
        String cacheFile = cacheDir + "/vkgraphics.shaderrefl." + m_signature + ".cache";
        if (cacheDir.empty())
        {
            spvReflectCreateShaderModule(compiledCode.size() * sizeof(u32), compiledCode.data(), &m_reflectModule);
            u32 numDescSets = 0;
            spvReflectEnumerateDescriptorSets(&m_reflectModule, &numDescSets, nullptr);
            m_reflectSets.resize(numDescSets);
            spvReflectEnumerateDescriptorSets(&m_reflectModule, &numDescSets, m_reflectSets.data());
        }
        else
        {
            // check if cache exists
            std::ifstream cache(cacheFile, std::ios::binary);
            if (cache.is_open())
            {
                cache.close();
                RecoverReflectionData();
            }
            else
            {
                spvReflectCreateShaderModule(moduleCI.codeSize, moduleCI.pCode, &m_reflectModule);
                u32 numDescSets = 0;
                spvReflectEnumerateDescriptorSets(&m_reflectModule, &numDescSets, nullptr);
                m_reflectSets.resize(numDescSets);
                spvReflectEnumerateDescriptorSets(&m_reflectModule, &numDescSets, m_reflectSets.data());
                m_reflectionCreated = true;
                CacheReflectionData();
            }
        }
    }

    IFRIT_APIDECL void ShaderModule::CacheReflectionData()
    {
        using Ifrit::SizeCast;
        // Currently, only writes the number of descriptor sets
        auto          cacheDir  = m_context->GetCacheDir();
        String        cacheFile = cacheDir + "/vkgraphics.shaderrefl." + m_signature + ".cache";
        std::ofstream cache(cacheFile, std::ios::binary);
        u32           numDescSets = SizeCast<u32>(m_reflectSets.size());
        cache.write(reinterpret_cast<const char*>(&numDescSets), sizeof(u32));
        cache.close();
    }

    IFRIT_APIDECL void ShaderModule::RecoverReflectionData()
    {
        auto          cacheDir  = m_context->GetCacheDir();
        String        cacheFile = cacheDir + "/vkgraphics.shaderrefl." + m_signature + ".cache";
        std::ifstream cache(cacheFile, std::ios::binary);
        if (!cache.is_open())
        {
            iError("Failed to open shader reflection cache file: {}", cacheFile);
            std::abort();
        }
        u32 numDescSets = 0;
        cache.read(reinterpret_cast<char*>(&numDescSets), sizeof(u32));

        m_reflectSets.resize(numDescSets);
        cache.close();
    }

    IFRIT_APIDECL ShaderModule::~ShaderModule()
    {
        vkDestroyShaderModule(m_context->GetDevice(), m_module, nullptr);
        m_module = VK_NULL_HANDLE;
        if (m_reflectionCreated)
        {
            spvReflectDestroyShaderModule(&m_reflectModule);
        }
    }

    IFRIT_APIDECL VkShaderModule                  ShaderModule::GetModule() const { return m_module; }

    IFRIT_APIDECL VkPipelineShaderStageCreateInfo ShaderModule::GetStageCI() const { return m_stageCI; }
} // namespace Ifrit::Graphics::VulkanGraphics
