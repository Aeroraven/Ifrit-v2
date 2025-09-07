
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

#include "ifrit/vkrhi/engine/vkrenderer/Command.h"
#include "ifrit/vkrhi/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkrhi/engine/vkrenderer/Pipeline.h"

#include "ifrit/vkrhi/engine/vkrenderer/Binding.h"

namespace Ifrit::RHI::VulkanAdapter
{

    class IFRIT_APIDECL HwRaytracingContext
    {
    private:
        EngineContext*                                  m_context;
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties;

    public:
        HwRaytracingContext(EngineContext* ctx);
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR getProperties() const;
        u32                                             GetShaderGroupHandleSize() const;
        u32                                             getAlignedShaderGroupHandleSize() const;
    };

    class IFRIT_APIDECL BottomLevelAS : public RHI::RhiRTInstance
    {
    private:
        VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;
        EngineContext*             m_context;
        Ref<SingleBuffer>          m_blasBuffer    = nullptr;
        Ref<SingleBuffer>          m_scratchBuffer = nullptr;
        RHI::RhiDeviceAddr         m_deviceAddress = 0;

    public:
        BottomLevelAS(EngineContext* ctx);
        void PrepareGeometryData(const Vec<RHI::RhiRTGeometryReference>& geometry, CommandBuffer* cmd);
        virtual RHI::RhiDeviceAddr GetDeviceAddress() const override;
    };

    class IFRIT_APIDECL TopLevelAS : public RHI::RhiRTScene
    {
    private:
        VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;
        EngineContext*             m_context;
        Ref<SingleBuffer>          m_tlasBuffer    = nullptr;
        Ref<SingleBuffer>          m_scratchBuffer = nullptr;
        RHI::RhiDeviceAddr         m_deviceAddress = 0;

    public:
        TopLevelAS(EngineContext* ctx);
        void                       PrepareInstanceData(const Vec<RHI::RhiRTInstance>& instances, CommandBuffer* cmd);
        virtual RHI::RhiDeviceAddr GetDeviceAddress() const override;
    };

    class IFRIT_APIDECL ShaderBindingTable : public RHI::RhiRTShaderBindingTable
    {
    private:
        EngineContext*                            m_context;
        HwRaytracingContext*                      m_rtContext;
        Ref<SingleBuffer>                         m_sbtBuffer = nullptr;

        Vec<Ref<SingleBuffer>>                    m_shaderBuffers;
        Vec<VkStridedDeviceAddressRegionKHR>      m_stridedRegions;
        Vec<const RHI::RhiShader*>                m_shaders;
        Vec<VkRayTracingShaderGroupCreateInfoKHR> m_shaderGroupsCI;
        Vec<u32>                                  m_numGroups;

    private:
        void appendShaderBindingTable(const Vec<RHI::RhiRTShaderGroup>& groups);

    public:
        ShaderBindingTable(EngineContext* ctx, HwRaytracingContext* rtContext);
        void                       PrepareShaderBindingTable(const Vec<Vec<RHI::RhiRTShaderGroup>>& groups);
        Vec<const RHI::RhiShader*> GetShaders() const;
        Vec<VkRayTracingShaderGroupCreateInfoKHR> GetShaderGroupsCI() const;
        Vec<VkStridedDeviceAddressRegionKHR>      GetStridedRegions() const;

        inline SingleBuffer*                      GetSbtBuffer(u32 index) { return m_shaderBuffers[index].get(); }

        inline Vec<u32>                           GetNumGroups() { return m_numGroups; }
    };

    struct RaytracePipelineCreateInfo
    {
        ShaderBindingTable*        sbt;
        Vec<VkDescriptorSetLayout> descriptorSetLayouts;
        u32                        pushConstSize = 0;
        u32                        maxRecursion  = 1;
    };

    class IFRIT_APIDECL RaytracingPipeline : public PipelineBase
    {
    public:
        RaytracePipelineCreateInfo m_createInfo;
        HwRaytracingContext*       m_rtContext;

    public:
        RaytracingPipeline(EngineContext* ctx, HwRaytracingContext* rtctx, const RaytracePipelineCreateInfo& ci)
            : PipelineBase(ctx), m_createInfo(ci), m_rtContext(rtctx)
        {
            Init();
        }

    protected:
        void Init();
    };

    class IFRIT_APIDECL RaytracingPipelineCache
    {
    private:
        EngineContext*                  m_context;
        HwRaytracingContext*            m_rtContext;

        Vec<Owner<RaytracingPipeline>>  m_raytracingPipelines;
        Vec<RaytracePipelineCreateInfo> m_raytracingPipelineCI;
        HashMap<u64, Vec<int>>          m_rtPipelineHash;

    public:
        RaytracingPipelineCache(EngineContext* ctx, HwRaytracingContext* rtctx) : m_context(ctx), m_rtContext(rtctx) {}
        RaytracingPipelineCache(const RaytracingPipelineCache& p)            = delete;
        RaytracingPipelineCache& operator=(const RaytracingPipelineCache& p) = delete;

        u64                      RaytracingPipelineHash(const RaytracePipelineCreateInfo& ci);
        bool RaytracingPipelineEqual(const RaytracePipelineCreateInfo& a, const RaytracePipelineCreateInfo& b);
        RaytracingPipeline* GetRaytracingPipeline(const RaytracePipelineCreateInfo& ci);
    };

    class IFRIT_APIDECL RaytracingPass : public RHI::RhiRTPass
    {
    private:
        EngineContext*                                  m_context;
        RaytracingPipeline*                             m_pipeline = nullptr;

        DescriptorManager*                              m_descriptorManager;
        RaytracingPipelineCache*                        m_pipelineCache;

        RHI::RhiRTShaderBindingTable*                   m_sbt                    = nullptr;
        u32                                             m_maxRecursion           = 1;
        u32                                             m_numBindlessDescriptors = 0;
        u32                                             m_pushConstSize          = 0;

        std::function<void(RHI::RhiRenderPassContext*)> m_recordFunc;

        u32                                             m_rayGenId   = ~0u;
        u32                                             m_missId     = ~0u;
        u32                                             m_hitGroupId = ~0u;
        u32                                             m_callableId = ~0u;

        u32                                             m_regionWidth  = 0;
        u32                                             m_regionHeight = 0;
        u32                                             m_regionDepth  = 0;

    public:
        RaytracingPass(
            EngineContext* context, DescriptorManager* descriptorManager, RaytracingPipelineCache* pipelineCache)
            : m_context(context), m_descriptorManager(descriptorManager), m_pipelineCache(pipelineCache)
        {
        }

        void SetShaderGroups(RHI::RhiRTShaderBindingTable* sbt);
        void SetMaxRecursion(u32 maxRecursion);
        void SetNumBindlessDescriptors(u32 numDescriptors);
        void SetPushConstSize(u32 size);
        void SetRecordFunction(std::function<void(RHI::RhiRenderPassContext*)> func);

        void SetTraceRegion(u32 width, u32 height, u32 depth);
        void SetShaderIds(u32 rayGen, u32 miss, u32 hitGroup, u32 callable);

    protected:
        void Build();

    public:
        void Run(const RHI::RhiCommandList* cmd);
    };

} // namespace Ifrit::RHI::VulkanAdapter
