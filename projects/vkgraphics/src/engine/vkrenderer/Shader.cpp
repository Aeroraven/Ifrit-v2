
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

// #include "spirv_reflect/spirv_reflect.c"

#include "ifrit/vkgraphics/engine/vkrenderer/Shader.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include "ifrit/core/algo/StlStringUtils.h"
#include <fstream>
#include <iostream>

#include "ifrit/shadercompile/helper/ShaderCompileHelper.h"

namespace Ifrit::RHI::VulkanAdapter
{

    IFRIT_APIDECL ShaderModule::ShaderModule(EngineContext* ctx, const ShaderModuleCI& ci)
    {
        m_context = ctx;
        VkShaderModuleCreateInfo moduleCI{};
        if (ci.stage == RHI::RhiShaderStage::Vertex)
        {
            m_stageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
        }
        else if (ci.stage == RHI::RhiShaderStage::Fragment)
        {
            m_stageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        else if (ci.stage == RHI::RhiShaderStage::Compute)
        {
            m_stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        else if (ci.stage == RHI::RhiShaderStage::Mesh)
        {
            m_stageCI.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        }
        else if (ci.stage == RHI::RhiShaderStage::Task)
        {
            m_stageCI.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        }
        // Raytracing
        else if (ci.stage == RHI::RhiShaderStage::RTRayGen)
        {
            m_stageCI.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        }
        else if (ci.stage == RHI::RhiShaderStage::RTClosestHit)
        {
            m_stageCI.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        }
        else if (ci.stage == RHI::RhiShaderStage::RTMiss)
        {
            m_stageCI.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        }
        else if (ci.stage == RHI::RhiShaderStage::RTAnyHit)
        {
            m_stageCI.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
        }
        else if (ci.stage == RHI::RhiShaderStage::RTIntersection)
        {
            m_stageCI.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
        }

        Vec<u32> compiledCode;
        VkDevice device   = m_context->GetDevice();
        moduleCI.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCI.pCode    = reinterpret_cast<const u32*>(ci.m_IRCode.data());
        moduleCI.codeSize = SizeCast<u32>(ci.m_IRCode.size() * sizeof(char));

        vkrVulkanAssert(vkCreateShaderModule(device, &moduleCI, nullptr, &m_module), "Failed to create shader module");
        if (m_context->IsDebugMode())
        {
            // add shader name
            VkDebugUtilsObjectNameInfoEXT nameInfo{};
            nameInfo.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
            nameInfo.objectType   = VK_OBJECT_TYPE_SHADER_MODULE;
            nameInfo.objectHandle = reinterpret_cast<u64>(m_module);

            String shaderName    = "IfShader." + ci.m_ShaderName;
            nameInfo.pObjectName = shaderName.c_str();

            auto extFunc = m_context->GetExtensionFunction();
            extFunc.p_vkSetDebugUtilsObjectNameEXT(device, &nameInfo);
        }

        m_stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

        m_ci             = ci;
        m_stageCI.module = m_module;
        m_stageCI.pName  = m_ci.m_EntryPoint.c_str();
        m_stageCI.flags  = 0;
        m_stageCI.pNext  = nullptr;
    }

    IFRIT_APIDECL void ShaderModule::CacheReflectionData()
    {
        using Ifrit::SizeCast;
        // Currently, only writes the number of descriptor sets
        auto          cacheDir  = m_context->GetCacheDir();
        String        cacheFile = cacheDir + "/vkgraphics.shaderrefl." + m_signature + ".cache";
        std::ofstream cache(cacheFile, std::ios::binary);
        u32           numDescSets = 0; // SizeCast<u32>(m_reflectSets.size());
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
        // m_reflectSets.resize(numDescSets);
        cache.close();
    }

    IFRIT_APIDECL ShaderModule::~ShaderModule()
    {
        vkDestroyShaderModule(m_context->GetDevice(), m_module, nullptr);
        m_module = VK_NULL_HANDLE;
        // if (m_reflectionCreated)
        // {
        //     spvReflectDestroyShaderModule(&m_reflectModule);
        // }
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
                    m_DefineIds[defineName] = SizeCast<u32>(m_DefineNames.size()) - 1;
                    m_MultiCompileIds.push_back(SizeCast<u32>(m_DefineNames.size()) - 1);
                }
                else if (tokens[1] == "ifrit.shader_feature")
                {
                    auto defineName = tokens[2];
                    m_DefineNames.push_back(defineName);
                    m_DefineIds[defineName] = SizeCast<u32>(m_DefineNames.size()) - 1;
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

        auto stageTranslate = [](RHI::RhiShaderStage stage) -> ShaderCompile::ShaderCompileStage {
            switch (stage)
            {
                case RHI::RhiShaderStage::Vertex:
                    return ShaderCompile::ShaderCompileStage::VertexShader;
                case RHI::RhiShaderStage::Fragment:
                    return ShaderCompile::ShaderCompileStage::FragmentShader;
                case RHI::RhiShaderStage::Compute:
                    return ShaderCompile::ShaderCompileStage::ComputeShader;
                case RHI::RhiShaderStage::Mesh:
                    return ShaderCompile::ShaderCompileStage::MeshShader;
                case RHI::RhiShaderStage::Task:
                    return ShaderCompile::ShaderCompileStage::AmplificationShader;
                default:
                    iError("Unsupported shader stage: {}", static_cast<u32>(stage));
                    std::abort();
                    return ShaderCompile::ShaderCompileStage::VertexShader; // Fallback
            }
        };

        auto sourceTypeConvert = [](RHI::RhiShaderSourceType sourceType) -> ShaderCompile::ShaderSourceFormat {
            switch (sourceType)
            {
                case RHI::RhiShaderSourceType::GLSLCode:
                    return ShaderCompile::ShaderSourceFormat::GLSL;
                case RHI::RhiShaderSourceType::SlangCode:
                    return ShaderCompile::ShaderSourceFormat::Slang;
                case RHI::RhiShaderSourceType::HLSLCode:
                    return ShaderCompile::ShaderSourceFormat::HLSL; // Fallback
                default:
                    iError("Unsupported shader source type: {}", static_cast<u32>(sourceType));
                    std::abort();
                    return ShaderCompile::ShaderSourceFormat::GLSL; // Fallback
            }
        };

        ShaderCompile::ShaderCompileJob job;
        HashMap<String, String>         definitionsInternal;
        job.m_Name            = m_CI.m_FileName;
        job.m_Source.m_Code   = String(m_CI.m_Code.begin(), m_CI.m_Code.end());
        job.m_Source.m_Format = sourceTypeConvert(m_CI.m_SourceType);
        job.m_EntryPoint      = m_CI.m_EntryPoint;
        job.m_Stage           = stageTranslate(m_CI.m_Stage);
        for (const auto& define : defines)
        {
            definitionsInternal[define] = "1"; // Set all defines to 1
        }
        job.m_Definitions = definitionsInternal;

        auto compiler = ShaderCompile::ShaderCompileHelper();
        if (job.m_Source.m_Format == ShaderCompile::ShaderSourceFormat::Slang)
        {
            compiler.SetIncludeBase(IFRIT_VKGRAPHICS_SHARED_SHADER_NEXT_INCLUDE_BASE);
        }
        else
        {
            compiler.SetIncludeBase(IFRIT_VKGRAPHICS_SHARED_SHADER_PATH);
        }

        compiler.SetCacheDir(m_Context->GetCacheDir());
        compiler.SetOptimization(ShaderCompile::ShaderCompileOptimization::Performance);

        auto           output = compiler.CompileShaderFromSource(job, ShaderCompile::ShaderIRFormat::SpirV);
        auto           irSize = output.m_IR.m_Data.GetSize();
        // iDebug("IR size: {} bytes", irSize);
        ShaderModuleCI shaderModuleCI;
        shaderModuleCI.m_IRCode     = output.m_IR.m_Data.ToString();
        shaderModuleCI.m_EntryPoint = "main"; // m_CI.m_EntryPoint;
        shaderModuleCI.stage        = m_CI.m_Stage;
        shaderModuleCI.m_ShaderName = m_CI.m_FileName;

        auto shaderModule            = MakeOwner<ShaderModule>(m_Context, shaderModuleCI);
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
        u64 retainedId = curPermId;
        PrecompileMultiCompileShadersImpl(curVariantTag + 1, curPermId);
        retainedId |= (1ull << m_MultiCompileIds[curVariantTag]);
        PrecompileMultiCompileShadersImpl(curVariantTag + 1, retainedId);
    }

    IFRIT_APIDECL RHI::RhiShader* ShaderCollection::GetVariant(const Vec<String>& defines)
    {
        u64 permId = 0;
        for (const auto& define : defines)
        {
            if (m_DefineIds.count(define) > 0)
            {
                auto id = m_DefineIds[define];
                permId |= (1ull << id);
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

} // namespace Ifrit::RHI::VulkanAdapter
