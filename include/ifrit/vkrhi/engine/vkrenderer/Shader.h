
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

#pragma once
#include "ifrit/vkrhi/common/Pch.h"
#include "ifrit/vkrhi/engine/vkrenderer/EngineContext.h"
#include "spirv_reflect/spirv_reflect.h"
#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanAdapter
{

    struct ShaderModuleCI
    {
        String              m_IRCode;
        RHI::RhiShaderStage stage;
        String              m_ShaderName;
        String              m_EntryPoint;
    };

    class IFRIT_APIDECL ShaderModule : public RHI::RhiShader
    {
    private:
        VkShaderModule                  m_module;
        VkPipelineShaderStageCreateInfo m_stageCI{};
        EngineContext*                  m_context;
        ShaderModuleCI                  m_ci;
        String                          m_entryPoint;

        // Intended for pipeline cache
        String                          m_signature;

    public:
        ShaderModule(EngineContext* ctx, const ShaderModuleCI& ci);
        ~ShaderModule();
        VkShaderModule                  GetModule() const;
        VkPipelineShaderStageCreateInfo GetStageCI() const;
        inline u32                      GetCodeSize() const { return SizeCast<u32>(m_ci.m_IRCode.size()); }
        inline u32                      GetNumDescriptorSets() const override
        {
            std::abort();
            return 0;
        }
        virtual RHI::RhiShaderStage GetStage() const override { return m_ci.stage; }

        void                        CacheReflectionData();
        void                        RecoverReflectionData();

        // get signature
        inline String               GetSignature() const { return m_signature; }
    };

    struct ShaderCollectionCI
    {
        Vec<char>                m_Code;
        String                   m_EntryPoint;
        RHI::RhiShaderStage      m_Stage;
        RHI::RhiShaderSourceType m_SourceType;
        String                   m_FileName;
    };

    class IFRIT_APIDECL ShaderCollection : public RHI::RhiShaderCollection, public NonCopyable
    {
    private:
        Vec<String>                     m_DefineNames;
        HashMap<String, u32>            m_DefineIds;
        HashMap<u64, Ref<ShaderModule>> m_ShaderVariants;
        EngineContext*                  m_Context;
        ShaderCollectionCI              m_CI;

        Vec<u32>                        m_MultiCompileIds;
        bool                            m_MultiCompileReady = false;

    private:
        void CompileShaderVariant(u64 permId);
        void PrecompileMultiCompileShaders();
        void PrecompileMultiCompileShadersImpl(u32 curVariantTag, u64 curPermId);

    public:
        ShaderCollection(EngineContext* ctx, const ShaderCollectionCI& ci);
        virtual ~ShaderCollection() = default;
        virtual RHI::RhiShader* GetVariant(const Vec<String>& defines) override;
        virtual bool            MultiCompileReady() override;
    };
} // namespace Ifrit::RHI::VulkanAdapter