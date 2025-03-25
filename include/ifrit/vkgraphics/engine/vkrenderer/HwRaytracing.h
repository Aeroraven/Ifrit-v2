
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include <memory>
#include <unordered_map>

#include "ifrit/vkgraphics/engine/vkrenderer/Command.h"
#include "ifrit/vkgraphics/engine/vkrenderer/MemoryResource.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Pipeline.h"

#include "ifrit/vkgraphics/engine/vkrenderer/Binding.h"

namespace Ifrit::GraphicsBackend::VulkanGraphics {

class IFRIT_APIDECL HwRaytracingContext {
private:
  EngineContext *m_context;
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties;

public:
  HwRaytracingContext(EngineContext *ctx);
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR getProperties() const;
  u32 getShaderGroupHandleSize() const;
  u32 getAlignedShaderGroupHandleSize() const;
};

class IFRIT_APIDECL BottomLevelAS : public Rhi::RhiRTInstance {
private:
  VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;
  EngineContext *m_context;
  std::shared_ptr<SingleBuffer> m_blasBuffer = nullptr;
  std::shared_ptr<SingleBuffer> m_scratchBuffer = nullptr;
  Rhi::RhiDeviceAddr m_deviceAddress = 0;

public:
  BottomLevelAS(EngineContext *ctx);
  void prepareGeometryData(const std::vector<Rhi::RhiRTGeometryReference> &geometry, CommandBuffer *cmd);
  virtual Rhi::RhiDeviceAddr getDeviceAddress() const override;
};

class IFRIT_APIDECL TopLevelAS : public Rhi::RhiRTScene {
private:
  VkAccelerationStructureKHR m_as = VK_NULL_HANDLE;
  EngineContext *m_context;
  std::shared_ptr<SingleBuffer> m_tlasBuffer = nullptr;
  std::shared_ptr<SingleBuffer> m_scratchBuffer = nullptr;
  Rhi::RhiDeviceAddr m_deviceAddress = 0;

public:
  TopLevelAS(EngineContext *ctx);
  void prepareInstanceData(const std::vector<Rhi::RhiRTInstance> &instances, CommandBuffer *cmd);
  virtual Rhi::RhiDeviceAddr getDeviceAddress() const override;
};

class IFRIT_APIDECL ShaderBindingTable : public Rhi::RhiRTShaderBindingTable {
private:
  EngineContext *m_context;
  HwRaytracingContext *m_rtContext;
  std::shared_ptr<SingleBuffer> m_sbtBuffer = nullptr;

  std::vector<std::shared_ptr<SingleBuffer>> m_shaderBuffers;
  std::vector<VkStridedDeviceAddressRegionKHR> m_stridedRegions;
  std::vector<const Rhi::RhiShader *> m_shaders;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_shaderGroupsCI;
  std::vector<u32> m_numGroups;

private:
  void appendShaderBindingTable(const std::vector<Rhi::RhiRTShaderGroup> &groups);

public:
  ShaderBindingTable(EngineContext *ctx, HwRaytracingContext *rtContext);
  void prepareShaderBindingTable(const std::vector<std::vector<Rhi::RhiRTShaderGroup>> &groups);
  std::vector<const Rhi::RhiShader *> getShaders() const;
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> getShaderGroupsCI() const;
  std::vector<VkStridedDeviceAddressRegionKHR> getStridedRegions() const;

  inline SingleBuffer *getSbtBuffer(u32 index) { return m_shaderBuffers[index].get(); }

  inline std::vector<u32> getNumGroups() { return m_numGroups; }
};

struct RaytracePipelineCreateInfo {
  ShaderBindingTable *sbt;
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  u32 pushConstSize = 0;
  u32 maxRecursion = 1;
};

class IFRIT_APIDECL RaytracingPipeline : public PipelineBase {
public:
  RaytracePipelineCreateInfo m_createInfo;
  HwRaytracingContext *m_rtContext;

public:
  RaytracingPipeline(EngineContext *ctx, HwRaytracingContext *rtctx, const RaytracePipelineCreateInfo &ci)
      : PipelineBase(ctx), m_createInfo(ci), m_rtContext(rtctx) {
    init();
  }

protected:
  void init();
};

class IFRIT_APIDECL RaytracingPipelineCache {
private:
  EngineContext *m_context;
  HwRaytracingContext *m_rtContext;

  std::vector<std::unique_ptr<RaytracingPipeline>> m_raytracingPipelines;
  std::vector<RaytracePipelineCreateInfo> m_raytracingPipelineCI;
  std::unordered_map<u64, std::vector<int>> m_rtPipelineHash;

public:
  RaytracingPipelineCache(EngineContext *ctx, HwRaytracingContext *rtctx) : m_context(ctx), m_rtContext(rtctx) {}
  RaytracingPipelineCache(const RaytracingPipelineCache &p) = delete;
  RaytracingPipelineCache &operator=(const RaytracingPipelineCache &p) = delete;

  u64 raytracingPipelineHash(const RaytracePipelineCreateInfo &ci);
  bool raytracingPipelineEqual(const RaytracePipelineCreateInfo &a, const RaytracePipelineCreateInfo &b);
  RaytracingPipeline *getRaytracingPipeline(const RaytracePipelineCreateInfo &ci);
};

class IFRIT_APIDECL RaytracingPass : public Rhi::RhiRTPass {
private:
  EngineContext *m_context;
  RaytracingPipeline *m_pipeline = nullptr;

  DescriptorManager *m_descriptorManager;
  RaytracingPipelineCache *m_pipelineCache;

  Rhi::RhiRTShaderBindingTable *m_sbt = nullptr;
  u32 m_maxRecursion = 1;
  u32 m_numBindlessDescriptors = 0;
  u32 m_pushConstSize = 0;

  std::function<void(Rhi::RhiRenderPassContext *)> m_recordFunc;

  u32 m_rayGenId = ~0u;
  u32 m_missId = ~0u;
  u32 m_hitGroupId = ~0u;
  u32 m_callableId = ~0u;

  u32 m_regionWidth = 0;
  u32 m_regionHeight = 0;
  u32 m_regionDepth = 0;

public:
  RaytracingPass(EngineContext *context, DescriptorManager *descriptorManager, RaytracingPipelineCache *pipelineCache)
      : m_context(context), m_descriptorManager(descriptorManager), m_pipelineCache(pipelineCache) {}

  void setShaderGroups(Rhi::RhiRTShaderBindingTable *sbt);
  void setMaxRecursion(u32 maxRecursion);
  void setNumBindlessDescriptors(u32 numDescriptors);
  void setPushConstSize(u32 size);
  void setRecordFunction(std::function<void(Rhi::RhiRenderPassContext *)> func);

  void setTraceRegion(u32 width, u32 height, u32 depth);
  void setShaderIds(u32 rayGen, u32 miss, u32 hitGroup, u32 callable);

protected:
  void build();

public:
  void run(const Rhi::RhiCommandList *cmd);
};

} // namespace Ifrit::GraphicsBackend::VulkanGraphics
