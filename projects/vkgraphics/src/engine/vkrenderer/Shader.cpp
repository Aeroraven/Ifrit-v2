
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

#include "ifrit/vkgraphics/engine/vkrenderer/Shader.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include "sha1/sha1.hpp"
#include "ifrit/core/algo/StlStringUtils.h"
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

    String PrecompileShaderFile(const String& source_name, shaderc_shader_kind kind, const String& source)
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

    Vec<u32> CompileShaderFile(
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
        m_ci.m_Permutations = ci.m_Permutations;
        if (ci.sourceType == Rhi::RhiShaderSourceType::GLSLCode)
        {

            auto   cacheDir = ctx->GetCacheDir();

            // Shader cache is a temporary solution,
            // PSO cache should be used in the future
            SHA1   sha1;
            String rawCode(ci.code.begin(), ci.code.end());
            // add glsl version to the shader code
            
            // If permutations are used, add defines to the shader code
            if (!ci.m_Permutations.empty())
            {
                for (const auto& perm : ci.m_Permutations)
                {
                    rawCode = "#define " + perm + " 1\n" + rawCode;
                }
            }

            rawCode = "#version 450\n" + rawCode;


            String precompiled;
            precompiled = PrecompileShaderFile(ci.fileName, static_cast<shaderc_shader_kind>(kind), rawCode);
            sha1.update(precompiled);
            auto hash   = sha1.final();
            m_signature = hash;

            if (cacheDir.empty())
            {
                compiledCode      = CompileShaderFile(ci.fileName, static_cast<shaderc_shader_kind>(kind), rawCode);
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
                    compiledCode = CompileShaderFile(ci.fileName, static_cast<shaderc_shader_kind>(kind), rawCode);
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
        if (m_context->IsDebugMode())
        {
            // add shader name
            VkDebugUtilsObjectNameInfoEXT nameInfo{};
            nameInfo.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            nameInfo.objectType   = VK_OBJECT_TYPE_SHADER_MODULE;
            nameInfo.objectHandle = reinterpret_cast<u64>(m_module);

            String shaderName = "IfShader." + ci.fileName + "(";
            for (auto& define : m_ci.m_Permutations)
            {
                shaderName += define + ",";
            }
            shaderName += ")";
            nameInfo.pObjectName = shaderName.c_str();

            auto extFunc = m_context->GetExtensionFunction();
            extFunc.p_vkSetDebugUtilsObjectNameEXT(device, &nameInfo);

            if (m_ci.m_Permutations.size() > 0)
            {
                printf("Shader %s with permutations %s\n", m_ci.fileName.c_str(),
                    JoinString(m_ci.m_Permutations, ",").c_str());
            }
        }

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

    // Shader collection
    IFRIT_APIDECL ShaderCollection::ShaderCollection(EngineContext* ctx, const ShaderCollectionCI& ci)
        : m_Context(ctx), m_CI(ci)
    {
        auto        codeStr = String(m_CI.m_Code.begin(), m_CI.m_Code.end());
        auto        defines = SplitString(codeStr, "\n");
        Vec<String> glslLines;
        for (const auto& define : defines)
        {
            if (define.starts_with("#pragma"))
            {
                auto tokens = SplitString(define, " ");
                if (tokens[1] == "ifrit.multi_compile")
                {
                    auto defineName = tokens[2];
                    m_DefineNames.push_back(defineName);
                    m_DefineIds[defineName] = m_DefineNames.size() - 1;
                    m_MultiCompileIds.push_back(m_DefineNames.size() - 1);
                }
                else if (tokens[1] == "ifrit.shader_feature")
                {
                    auto defineName = tokens[2];
                    m_DefineNames.push_back(defineName);
                    m_DefineIds[defineName] = m_DefineNames.size() - 1;
                }
            }
            else
            {
                glslLines.push_back(define);
            }
        }
        String glslCode = JoinString(glslLines, "\n");
        m_CI.m_Code     = Vec<char>(glslCode.begin(), glslCode.end());

        PrecompileMultiCompileShaders();
    }

    IFRIT_APIDECL void ShaderCollection::CompileShaderVariant(u64 permId)
    {
        if (m_ShaderVariants.count(permId) > 0)
        {
            return;
        }

        Vec<String> defines;
        auto        permIdCopy = permId;
        while (permId)
        {
            u32 trailingBit = Math::CountTrailingZero(permId);
            permId &= ~(1 << trailingBit);
            defines.push_back(m_DefineNames[trailingBit]);
        }

        ShaderModuleCI shaderModuleCI;
        shaderModuleCI.code           = m_CI.m_Code;
        shaderModuleCI.entryPoint     = m_CI.m_EntryPoint;
        shaderModuleCI.stage          = m_CI.m_Stage;
        shaderModuleCI.sourceType     = m_CI.m_SourceType;
        shaderModuleCI.fileName       = m_CI.m_FileName;
        shaderModuleCI.m_Permutations = std::move(defines);

        auto shaderModule            = std::make_unique<ShaderModule>(m_Context, shaderModuleCI);
        m_ShaderVariants[permIdCopy] = std::move(shaderModule);
    }

    IFRIT_APIDECL void ShaderCollection::PrecompileMultiCompileShaders()
    {
        if (m_MultiCompileIds.size() >= 12)
        {
            iError("Multi compile shaders are limited to 12 permutations, please reduce the number of permutations.");
            return;
        }
        PrecompileMultiCompileShadersImpl(0, 0);
        m_MultiCompileReady = true;
    }

    IFRIT_APIDECL void ShaderCollection::PrecompileMultiCompileShadersImpl(u32 curVariantTag, u64 curPermId)
    {
        if (curVariantTag == m_MultiCompileIds.size())
        {
            CompileShaderVariant(curPermId);
            return;
        }
        u32 retainedId = curPermId;
        PrecompileMultiCompileShadersImpl(curVariantTag + 1, curPermId);
        retainedId |= (1 << m_MultiCompileIds[curVariantTag]);
        PrecompileMultiCompileShadersImpl(curVariantTag + 1, retainedId);
    }

    IFRIT_APIDECL Rhi::RhiShader* ShaderCollection::GetVariant(const Vec<String>& defines)
    {
        u64 permId = 0;
        for (const auto& define : defines)
        {
            if (m_DefineIds.count(define) > 0)
            {
                auto id = m_DefineIds[define];
                permId |= (1 << id);
            }
            else
            {
                iError("Shader define {} not found in shader collection {}", define, m_CI.m_FileName);
                std::abort();
            }
        }
        CompileShaderVariant(permId);
        if (m_ShaderVariants.count(permId) > 0)
        {
            if (permId != 0)
            {
                // std::abort();
            }
            return m_ShaderVariants[permId].get();
        }
        else
        {
            iError("Shader variant {} not found in shader collection {}", permId, m_CI.m_FileName);
            std::abort();
        }
    }
    IFRIT_APIDECL bool ShaderCollection::MultiCompileReady() { return m_MultiCompileReady; }

} // namespace Ifrit::Graphics::VulkanGraphics
