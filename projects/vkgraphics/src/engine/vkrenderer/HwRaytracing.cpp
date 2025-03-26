#include "ifrit/vkgraphics/engine/vkrenderer/HwRaytracing.h"
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/vkgraphics/utility/Logger.h"
#include <numeric>

namespace Ifrit::Graphics::VulkanGraphics
{

    IFRIT_APIDECL HwRaytracingContext::HwRaytracingContext(EngineContext* ctx)
    {
        m_context                               = ctx;
        VkPhysicalDeviceProperties2 properties2 = {};
        properties2.sType                       = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        m_rtProperties.sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        properties2.pNext                       = &m_rtProperties;
        vkGetPhysicalDeviceProperties2(ctx->GetPhysicalDevice(), &properties2);
    }

    IFRIT_APIDECL VkPhysicalDeviceRayTracingPipelinePropertiesKHR HwRaytracingContext::getProperties() const
    {
        return m_rtProperties;
    }

    IFRIT_APIDECL u32 HwRaytracingContext::GetShaderGroupHandleSize() const
    {
        return m_rtProperties.shaderGroupHandleSize;
    }

    IFRIT_APIDECL u32 HwRaytracingContext::getAlignedShaderGroupHandleSize() const
    {
        return Math::alignUp(m_rtProperties.shaderGroupHandleSize, m_rtProperties.shaderGroupHandleAlignment);
    }

    IFRIT_APIDECL BottomLevelAS::BottomLevelAS(EngineContext* ctx)
    {
        m_context = ctx;
    }
    IFRIT_APIDECL void BottomLevelAS::PrepareGeometryData(const Vec<Rhi::RhiRTGeometryReference>& geometry,
        CommandBuffer*                                                                            cmd)
    {
        // TODO
        Vec<VkAccelerationStructureGeometryKHR>        geometries;
        Vec<VkAccelerationStructureBuildRangeInfoKHR>  BuildRanges;
        Vec<VkAccelerationStructureBuildRangeInfoKHR*> pBuildRanges;
        Vec<u32>                                       primitiveCounts;
        for (const auto& geo : geometry)
        {
            VkAccelerationStructureGeometryKHR geometryInfo = {};
            geometryInfo.sType                              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometryInfo.geometryType                       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            geometryInfo.flags                              = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geometryInfo.geometry.triangles.sType           = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            if (geo.m_vertexComponents == 3)
            {
                geometryInfo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            }
            else if (geo.m_vertexComponents == 4)
            {
                geometryInfo.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
            }
            geometryInfo.geometry.triangles.vertexData.deviceAddress    = geo.m_vertex;
            geometryInfo.geometry.triangles.vertexStride                = geo.m_vertexStride;
            geometryInfo.geometry.triangles.maxVertex                   = geo.m_numVertices;
            geometryInfo.geometry.triangles.indexType                   = VK_INDEX_TYPE_UINT32;
            geometryInfo.geometry.triangles.indexData.deviceAddress     = geo.m_index;
            geometryInfo.geometry.triangles.transformData.deviceAddress = geo.m_transform;
            geometries.push_back(geometryInfo);

            primitiveCounts.push_back(geo.m_numIndices / 3);

            // Build ranges
            VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
            buildRange.primitiveCount                           = geo.m_numIndices / 3;
            buildRange.firstVertex                              = 0;
            buildRange.primitiveOffset                          = 0;
            buildRange.transformOffset                          = 0;
            BuildRanges.push_back(buildRange);
        }

        // Build range pointers
        for (auto& range : BuildRanges)
        {
            pBuildRanges.push_back(&range);
        }

        // Prepare BLAS Info
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
        buildInfo.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type                                        = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags                                       = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.geometryCount                               = geometries.size();
        buildInfo.pGeometries                                 = geometries.data();

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
        auto                                     pfns     = m_context->GetExtensionFunction();
        pfns.p_vkGetAccelerationStructureBuildSizesKHR(m_context->GetDevice(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo,
            primitiveCounts.data(), &sizeInfo);

        // Create buffer
        BufferCreateInfo blasBufferCI;
        blasBufferCI.size        = sizeInfo.accelerationStructureSize;
        blasBufferCI.hostVisible = false;
        blasBufferCI.usage =
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        m_blasBuffer = std::make_shared<SingleBuffer>(m_context, blasBufferCI);

        // Create BLAS
        VkAccelerationStructureCreateInfoKHR asCI = {};
        asCI.sType                                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer                               = m_blasBuffer->GetBuffer();
        asCI.size                                 = sizeInfo.accelerationStructureSize;
        asCI.type                                 = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        vkrVulkanAssert(pfns.p_vkCreateAccelerationStructureKHR(m_context->GetDevice(), &asCI, nullptr, &m_as),
            "Failed to create BLAS");

        // Build BLAS
        BufferCreateInfo scratchBufferCI;
        scratchBufferCI.size        = sizeInfo.buildScratchSize;
        scratchBufferCI.hostVisible = false;
        scratchBufferCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        m_scratchBuffer             = std::make_shared<SingleBuffer>(m_context, scratchBufferCI);

        buildInfo.dstAccelerationStructure  = m_as;
        buildInfo.scratchData.deviceAddress = m_scratchBuffer->GetDeviceAddress();
        buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

        pfns.p_vkCmdBuildAccelerationStructuresKHR(cmd->GetCommandBuffer(), 1, &buildInfo, pBuildRanges.data());

        // Get device address
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
        addressInfo.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure                       = m_as;
        m_deviceAddress                                         = pfns.p_vkGetAccelerationStructureDeviceAddressKHR(m_context->GetDevice(), &addressInfo);
    }

    IFRIT_APIDECL Rhi::RhiDeviceAddr BottomLevelAS::GetDeviceAddress() const
    {
        return m_deviceAddress;
    }

    IFRIT_APIDECL TopLevelAS::TopLevelAS(EngineContext* ctx)
    {
        m_context = ctx;
    }

    IFRIT_APIDECL void TopLevelAS::PrepareInstanceData(const Vec<Rhi::RhiRTInstance>& instances,
        CommandBuffer*                                                                cmd)
    {
        // TODO
        VkTransformMatrixKHR                           transformMatrix = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f };
        Vec<VkAccelerationStructureInstanceKHR>        instanceData;
        Vec<VkAccelerationStructureGeometryKHR>        geometries;
        Vec<VkAccelerationStructureBuildRangeInfoKHR>  BuildRanges;
        Vec<VkAccelerationStructureBuildRangeInfoKHR*> pBuildRanges;
        for (auto i = 0; const auto& inst : instances)
        {
            VkAccelerationStructureInstanceKHR instance     = {};
            instance.transform                              = transformMatrix;
            instance.instanceCustomIndex                    = i++;
            instance.mask                                   = 0xFF;
            instance.instanceShaderBindingTableRecordOffset = 0;
            instance.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            instance.accelerationStructureReference         = inst.GetDeviceAddress();
            instanceData.push_back(instance);

            VkAccelerationStructureGeometryKHR geometryInfo    = {};
            geometryInfo.sType                                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            geometryInfo.geometryType                          = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            geometryInfo.flags                                 = VK_GEOMETRY_OPAQUE_BIT_KHR;
            geometryInfo.geometry.instances.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            geometryInfo.geometry.instances.arrayOfPointers    = VK_FALSE;
            geometryInfo.geometry.instances.data.deviceAddress = inst.GetDeviceAddress();
            geometries.push_back(geometryInfo);

            VkAccelerationStructureBuildRangeInfoKHR buildRange = {};
            buildRange.primitiveCount                           = 1;
            buildRange.firstVertex                              = 0;
            buildRange.primitiveOffset                          = 0;
            BuildRanges.push_back(buildRange);
        }

        // Build range pointers
        for (auto& range : BuildRanges)
        {
            pBuildRanges.push_back(&range);
        }

        u32                                         numInstances = instances.size();
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo    = {};
        buildInfo.sType                                          = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type                                           = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        buildInfo.flags                                          = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.geometryCount                                  = numInstances;
        buildInfo.pGeometries                                    = geometries.data();

        auto                                     pfns = m_context->GetExtensionFunction();

        VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
        sizeInfo.sType                                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        pfns.p_vkGetAccelerationStructureBuildSizesKHR(
            m_context->GetDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &numInstances, &sizeInfo);

        // Create As buffer
        BufferCreateInfo tlasBufferCI;
        tlasBufferCI.size        = sizeInfo.accelerationStructureSize;
        tlasBufferCI.hostVisible = false;
        tlasBufferCI.usage =
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        m_tlasBuffer = std::make_shared<SingleBuffer>(m_context, tlasBufferCI);

        // Create TLAS
        VkAccelerationStructureCreateInfoKHR asCI = {};
        asCI.sType                                = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer                               = m_tlasBuffer->GetBuffer();
        asCI.size                                 = sizeInfo.accelerationStructureSize;
        asCI.type                                 = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        vkrVulkanAssert(pfns.p_vkCreateAccelerationStructureKHR(m_context->GetDevice(), &asCI, nullptr, &m_as),
            "Failed to create TLAS");

        // Build TLAS
        BufferCreateInfo scratchBufferCI;
        scratchBufferCI.size        = sizeInfo.buildScratchSize;
        scratchBufferCI.hostVisible = false;
        scratchBufferCI.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        m_scratchBuffer             = std::make_shared<SingleBuffer>(m_context, scratchBufferCI);

        buildInfo.dstAccelerationStructure  = m_as;
        buildInfo.scratchData.deviceAddress = m_scratchBuffer->GetDeviceAddress();
        buildInfo.mode                      = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;

        pfns.p_vkCmdBuildAccelerationStructuresKHR(cmd->GetCommandBuffer(), 1, &buildInfo, pBuildRanges.data());

        // Get device address
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo = {};
        addressInfo.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure                       = m_as;

        m_deviceAddress = pfns.p_vkGetAccelerationStructureDeviceAddressKHR(m_context->GetDevice(), &addressInfo);
    }

    IFRIT_APIDECL Rhi::RhiDeviceAddr TopLevelAS::GetDeviceAddress() const
    {
        return m_deviceAddress;
    }

    IFRIT_APIDECL
    ShaderBindingTable::ShaderBindingTable(EngineContext* ctx, HwRaytracingContext* rtContext)
        : m_context(ctx), m_rtContext(rtContext) {}

    IFRIT_APIDECL void ShaderBindingTable::appendShaderBindingTable(const Vec<Rhi::RhiRTShaderGroup>& groups)
    {

        auto             numShaders = groups.size();
        BufferCreateInfo sbtBufferCI{};
        sbtBufferCI.size                          = numShaders * m_rtContext->getAlignedShaderGroupHandleSize();
        sbtBufferCI.hostVisible                   = true;
        sbtBufferCI.usage                         = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto                            sbtBuffer = std::make_shared<SingleBuffer>(m_context, sbtBufferCI);

        VkStridedDeviceAddressRegionKHR stridedRegion = {};
        stridedRegion.deviceAddress                   = sbtBuffer->GetDeviceAddress();
        stridedRegion.size                            = sbtBufferCI.size;
        stridedRegion.stride                          = m_rtContext->getAlignedShaderGroupHandleSize();

        m_shaderBuffers.push_back(sbtBuffer);
        m_stridedRegions.push_back(stridedRegion);

        sbtBuffer->MapMemory();
    }

    IFRIT_APIDECL void
    ShaderBindingTable::PrepareShaderBindingTable(const Vec<Vec<Rhi::RhiRTShaderGroup>>& groups)
    {
        std::unordered_map<const Rhi::RhiShader*, u32> shaderMap;

        auto                                           insertShader = [&](const Rhi::RhiShader* shader) {
            if (shader == nullptr)
            {
                return;
            }
            if (shaderMap.find(shader) == shaderMap.end())
            {
                shaderMap[shader] = m_shaders.size();
                m_shaders.push_back(shader);
            }
        };

        for (const auto& group : groups)
        {
            for (const auto& shader : group)
            {
                insertShader(shader.m_generalShader);
                insertShader(shader.m_closestHitShader);
                insertShader(shader.m_anyHitShader);
                insertShader(shader.m_intersectionShader);
            }
        }

        auto handleSize        = m_rtContext->GetShaderGroupHandleSize();
        auto alignedHandleSize = m_rtContext->getAlignedShaderGroupHandleSize();
        auto groupCount        = groups.size();
        auto sbtSize           = groupCount * alignedHandleSize;

        // Create binding table
        for (const auto& group : groups)
        {
            appendShaderBindingTable(group);
        }

        // Create group CIs
        auto lookupShaderIndex = [&](const Rhi::RhiShader* shader) -> u32 {
            if (shader == nullptr)
            {
                return VK_SHADER_UNUSED_KHR;
            }
            return shaderMap[shader];
        };

        for (auto& group : groups)
        {
            VkRayTracingShaderGroupCreateInfoKHR groupCI = {};
            groupCI.sType                                = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
            bool hasHitShaders                           = false;
            u32  numGroups                               = group.size();
            for (auto& shader : group)
            {
                hasHitShaders = false;
                if (shader.m_closestHitShader != nullptr || shader.m_anyHitShader != nullptr || shader.m_intersectionShader != nullptr)
                {
                    hasHitShaders = true;
                }
                if (hasHitShaders)
                {
                    groupCI.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
                }
                else
                {
                    groupCI.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
                }
                groupCI.generalShader      = lookupShaderIndex(shader.m_generalShader);
                groupCI.closestHitShader   = lookupShaderIndex(shader.m_closestHitShader);
                groupCI.anyHitShader       = lookupShaderIndex(shader.m_anyHitShader);
                groupCI.intersectionShader = lookupShaderIndex(shader.m_intersectionShader);

                m_shaderGroupsCI.push_back(groupCI);
            }
            m_numGroups.push_back(numGroups);
        }
    }

    IFRIT_APIDECL Vec<const Rhi::RhiShader*> ShaderBindingTable::GetShaders() const
    {
        return m_shaders;
    }

    IFRIT_APIDECL Vec<VkRayTracingShaderGroupCreateInfoKHR> ShaderBindingTable::GetShaderGroupsCI() const
    {
        return m_shaderGroupsCI;
    }

    IFRIT_APIDECL Vec<VkStridedDeviceAddressRegionKHR> ShaderBindingTable::GetStridedRegions() const
    {
        return m_stridedRegions;
    }

    IFRIT_APIDECL void RaytracingPipeline::Init()
    {
        using namespace Common::Utility;
        auto                              pfns = m_context->GetExtensionFunction();

        VkRayTracingPipelineCreateInfoKHR pipelineCI = {};
        pipelineCI.sType                             = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;

        auto                                 shaders = m_createInfo.sbt->GetShaders();
        Vec<VkPipelineShaderStageCreateInfo> stages;
        for (auto shader : shaders)
        {
            auto vshader = CheckedCast<const ShaderModule>(shader);
            stages.push_back(vshader->GetStageCI());
        }
        pipelineCI.pStages    = stages.data();
        pipelineCI.stageCount = SizeCast<u32>(stages.size());

        auto shaderGroups = m_createInfo.sbt->GetShaderGroupsCI();

        pipelineCI.pGroups    = shaderGroups.data();
        pipelineCI.groupCount = SizeCast<u32>(shaderGroups.size());

        pipelineCI.maxPipelineRayRecursionDepth = m_createInfo.maxRecursion;

        // Create layout
        VkPipelineLayoutCreateInfo layoutCI   = {};
        layoutCI.sType                        = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layoutCI.setLayoutCount               = SizeCast<u32>(m_createInfo.descriptorSetLayouts.size());
        layoutCI.pSetLayouts                  = m_createInfo.descriptorSetLayouts.data();
        VkPushConstantRange pushConstantRange = {};
        if (m_createInfo.pushConstSize > 0)
        {
            pushConstantRange.offset        = 0;
            pushConstantRange.size          = m_createInfo.pushConstSize;
            pushConstantRange.stageFlags    = VK_SHADER_STAGE_ALL;
            layoutCI.pushConstantRangeCount = 1;
            layoutCI.pPushConstantRanges    = &pushConstantRange;
        }

        vkrVulkanAssert(vkCreatePipelineLayout(m_context->GetDevice(), &layoutCI, nullptr, &m_layout),
            "Failed to create pipeline layout");

        // Create pipeline
        vkrVulkanAssert(pfns.p_vkCreateRayTracingPipelinesKHR(m_context->GetDevice(), VK_NULL_HANDLE, VK_NULL_HANDLE, 1,
                            &pipelineCI, nullptr, &m_pipeline),
            "Failed to create raytracing pipeline");

        // Prepare handles
        auto handleSize        = m_rtContext->GetShaderGroupHandleSize();
        auto alignedHandleSize = m_rtContext->getAlignedShaderGroupHandleSize();
        auto numGroups         = m_createInfo.sbt->GetNumGroups();

        auto totalGroups = 0;
        // accumulating the number of groups
        totalGroups = std::accumulate(numGroups.begin(), numGroups.end(), 0);

        auto      sbtHandleSize = totalGroups * alignedHandleSize;

        Vec<char> shaderGroupHandles(sbtHandleSize);
        pfns.p_vkGetRayTracingShaderGroupHandlesKHR(m_context->GetDevice(), m_pipeline, 0,
            m_rtContext->getAlignedShaderGroupHandleSize(), sbtHandleSize,
            shaderGroupHandles.data());

        auto curOffset = 0;
        for (auto i = 0; i < numGroups.size(); i++)
        {
            auto numGroup  = numGroups[i];
            auto groupSize = numGroup * alignedHandleSize;
            auto sbtBuffer = m_createInfo.sbt->GetSbtBuffer(i);
            sbtBuffer->MapMemory();
            sbtBuffer->WriteBuffer(shaderGroupHandles.data() + curOffset, groupSize, 0);
            curOffset += groupSize;
        }
    }

    IFRIT_APIDECL uint64_t RaytracingPipelineCache::RaytracingPipelineHash(const RaytracePipelineCreateInfo& ci)
    {
        using namespace Common::Utility;
        if (ci.sbt == nullptr)
        {
            vkrError("Shader binding table is null");
        }
        u64                 hash = 0x9e3779bf;
        std::hash<uint64_t> hashFunc;
        auto                shaders      = ci.sbt->GetShaders();
        auto                shaderGroups = ci.sbt->GetShaderGroupsCI();
        for (auto shader : shaders)
        {
            auto vshader = CheckedCast<const ShaderModule>(shader);
            hash ^= hashFunc(reinterpret_cast<u64>(vshader));
        }
        for (auto& group : shaderGroups)
        {
            hash ^= hashFunc(group.type);
            hash ^= hashFunc(group.generalShader);
            hash ^= hashFunc(group.closestHitShader);
            hash ^= hashFunc(group.anyHitShader);
            hash ^= hashFunc(group.intersectionShader);
        }
        hash ^= hashFunc(ci.pushConstSize);
        hash ^= hashFunc(ci.maxRecursion);
        for (auto& layout : ci.descriptorSetLayouts)
        {
            hash ^= hashFunc(reinterpret_cast<u64>(layout));
        }
        return hash;
    }

    IFRIT_APIDECL bool RaytracingPipelineCache::RaytracingPipelineEqual(const RaytracePipelineCreateInfo& a,
        const RaytracePipelineCreateInfo&                                                                 b)
    {
        if (a.sbt->GetShaders() != b.sbt->GetShaders())
        {
            return false;
        }
        if (a.descriptorSetLayouts != b.descriptorSetLayouts)
        {
            return false;
        }
        if (a.pushConstSize != b.pushConstSize)
        {
            return false;
        }
        if (a.maxRecursion != b.maxRecursion)
        {
            return false;
        }
        auto shaderGroupCIA = a.sbt->GetShaderGroupsCI();
        auto shaderGroupCIB = b.sbt->GetShaderGroupsCI();
        if (shaderGroupCIA.size() != shaderGroupCIB.size())
        {
            return false;
        }
        for (auto i = 0; i < shaderGroupCIA.size(); i++)
        {
            if (shaderGroupCIA[i].type != shaderGroupCIB[i].type)
            {
                return false;
            }
            if (shaderGroupCIA[i].generalShader != shaderGroupCIB[i].generalShader)
            {
                return false;
            }
            if (shaderGroupCIA[i].closestHitShader != shaderGroupCIB[i].closestHitShader)
            {
                return false;
            }
            if (shaderGroupCIA[i].anyHitShader != shaderGroupCIB[i].anyHitShader)
            {
                return false;
            }
            if (shaderGroupCIA[i].intersectionShader != shaderGroupCIB[i].intersectionShader)
            {
                return false;
            }
        }
        return true;
    }

    IFRIT_APIDECL RaytracingPipeline* RaytracingPipelineCache::GetRaytracingPipeline(const RaytracePipelineCreateInfo& ci)
    {
        auto hash = RaytracingPipelineHash(ci);
        if (m_rtPipelineHash.find(hash) != m_rtPipelineHash.end())
        {
            auto& pipelineIndices = m_rtPipelineHash[hash];
            for (auto& index : pipelineIndices)
            {
                if (RaytracingPipelineEqual(m_raytracingPipelineCI[index], ci))
                {
                    return m_raytracingPipelines[index].get();
                }
            }
        }

        auto pipeline = std::make_unique<RaytracingPipeline>(m_context, m_rtContext, ci);
        m_raytracingPipelines.push_back(std::move(pipeline));
        m_raytracingPipelineCI.push_back(ci);
        m_rtPipelineHash[hash].push_back(m_raytracingPipelines.size() - 1);
        return pipeline.get();
    }

    // RaytracingPass

    IFRIT_APIDECL void RaytracingPass::SetMaxRecursion(u32 maxRecursion)
    {
        m_maxRecursion = maxRecursion;
    }

    IFRIT_APIDECL void RaytracingPass::SetNumBindlessDescriptors(u32 numDescriptors)
    {
        m_numBindlessDescriptors = numDescriptors;
    }

    IFRIT_APIDECL void RaytracingPass::SetPushConstSize(u32 size)
    {
        m_pushConstSize = size;
    }

    IFRIT_APIDECL void RaytracingPass::SetShaderGroups(Rhi::RhiRTShaderBindingTable* sbt)
    {
        m_sbt = sbt;
    }

    IFRIT_APIDECL void RaytracingPass::SetTraceRegion(u32 width, u32 height, u32 depth)
    {
        m_regionWidth  = width;
        m_regionHeight = height;
        m_regionDepth  = depth;
    }

    IFRIT_APIDECL void RaytracingPass::SetRecordFunction(std::function<void(Rhi::RhiRenderPassContext*)> func)
    {
        m_recordFunc = func;
    }

    IFRIT_APIDECL void RaytracingPass::SetShaderIds(u32 rayGen, u32 miss, u32 hitGroup, u32 callable)
    {
        m_rayGenId   = rayGen;
        m_missId     = miss;
        m_hitGroupId = hitGroup;
        m_callableId = callable;
    }

    IFRIT_APIDECL void RaytracingPass::Build()
    {
        using namespace Common::Utility;

        RaytracePipelineCreateInfo ci;
        ci.sbt           = CheckedCast<ShaderBindingTable>(m_sbt);
        ci.maxRecursion  = m_maxRecursion;
        ci.pushConstSize = m_pushConstSize;
        for (int i = 0; i < m_numBindlessDescriptors; i++)
        {
            ci.descriptorSetLayouts.push_back(m_descriptorManager->GetParameterDescriptorSetLayout());
        }

        m_pipeline = m_pipelineCache->GetRaytracingPipeline(ci);
    }

    IFRIT_APIDECL void RaytracingPass::Run(const Rhi::RhiCommandList* cmd)
    {
        auto pfns = m_context->GetExtensionFunction();
        using namespace Common::Utility;
        if (m_pipeline == nullptr)
        {
            Build();
        }
        auto cmdraw = CheckedCast<CommandBuffer>(cmd);
        auto cmdx   = cmdraw->GetCommandBuffer();

        auto bindlessSet = m_descriptorManager->GetBindlessSet();
        vkCmdBindDescriptorSets(cmdx, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline->GetLayout(), 0, 1, &bindlessSet, 0,
            nullptr);
        vkCmdBindPipeline(cmdx, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline->GetPipeline());
        Rhi::RhiRenderPassContext m_passContext;
        m_passContext.m_cmd   = cmdraw;
        m_passContext.m_frame = 0;
        if (m_recordFunc)
        {
            m_recordFunc(&m_passContext);
        }
        // Trace Rays
        auto                            sbt            = CheckedCast<ShaderBindingTable>(m_sbt);
        auto                            stridedRegions = sbt->GetStridedRegions();

        VkStridedDeviceAddressRegionKHR emptyRegion = {};

        auto                            getStridedRegion = [&](u32 index) -> VkStridedDeviceAddressRegionKHR {
            if (index == ~0u)
            {
                return emptyRegion;
            }
            return stridedRegions[index];
        };
        auto raygenRegion   = getStridedRegion(m_rayGenId);
        auto missRegion     = getStridedRegion(m_missId);
        auto hitGroupRegion = getStridedRegion(m_hitGroupId);
        auto callableRegion = getStridedRegion(m_callableId);

        pfns.p_vkCmdTraceRaysKHR(cmdx, &raygenRegion, &missRegion, &hitGroupRegion, &callableRegion, m_regionWidth,
            m_regionHeight, m_regionDepth);
    }

} // namespace Ifrit::Graphics::VulkanGraphics