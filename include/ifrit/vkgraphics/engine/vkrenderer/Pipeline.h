#pragma once
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/vkgraphics/engine/vkrenderer/EngineContext.h"
#include "ifrit/vkgraphics/engine/vkrenderer/Shader.h"
#include <memory>

namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics {

struct GraphicsPipelineCreateInfo {
  uint32_t viewportCount;
  uint32_t scissorCount;
  Rhi::RhiRasterizerTopology topology;
  std::vector<VkFormat> colorAttachmentFormats;
  VkFormat depthAttachmentFormat;
  VkFormat stencilAttachmentFormat;
  std::vector<ShaderModule *> shaderModules;
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
  Rhi::RhiGeometryGenerationType geomGenType =
      Rhi::RhiGeometryGenerationType::Conventional;
};
struct ComputePipelineCreateInfo {
  ShaderModule *shaderModules;
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
};

class IFRIT_APIDECL PipelineBase {
protected:
  EngineContext *m_context;
  VkPipeline m_pipeline;
  VkPipelineLayout m_layout;
  bool m_layoutCreated = false;
  bool m_pipelineCreated = false;

public:
  PipelineBase(EngineContext *ctx) : m_context(ctx) {}
  virtual ~PipelineBase() {}
  inline VkPipeline getPipeline() const { return m_pipeline; }
  inline VkPipelineLayout getLayout() const { return m_layout; }
};

class IFRIT_APIDECL GraphicsPipeline : public PipelineBase {
private:
  GraphicsPipelineCreateInfo m_createInfo;

protected:
  void init();

public:
  GraphicsPipeline(EngineContext *ctx, const GraphicsPipelineCreateInfo &ci)
      : PipelineBase(ctx), m_createInfo(ci) {
    init();
  }
  virtual ~GraphicsPipeline();
};

class IFRIT_APIDECL ComputePipeline : public PipelineBase {
private:
  ComputePipelineCreateInfo m_createInfo;

protected:
  void init();

public:
  ComputePipeline(EngineContext *ctx, const ComputePipelineCreateInfo &ci)
      : PipelineBase(ctx), m_createInfo(ci) {
    init();
  }
  virtual ~ComputePipeline();
};

// Reuse pipelines that share the same layout
class IFRIT_APIDECL PipelineCache {
private:
  EngineContext *m_context;
  std::vector<std::unique_ptr<GraphicsPipeline>> m_graphicsPipelines;
  std::vector<GraphicsPipelineCreateInfo> m_graphicsPipelineCI;
  std::unordered_map<uint64_t, std::vector<int>> m_graphicsPipelineMap;

  std::vector<std::unique_ptr<ComputePipeline>> m_computePipelines;
  std::vector<ComputePipelineCreateInfo> m_computePipelineCI;
  std::unordered_map<uint64_t, std::vector<int>> m_computePipelineMap;

public:
  PipelineCache(EngineContext *context);
  PipelineCache(const PipelineCache &p) = delete;
  PipelineCache &operator=(const PipelineCache &p) = delete;

  uint64_t graphicsPipelineHash(const GraphicsPipelineCreateInfo &ci);
  bool graphicsPipelineEqual(const GraphicsPipelineCreateInfo &a,
                             const GraphicsPipelineCreateInfo &b);
  GraphicsPipeline *getGraphicsPipeline(const GraphicsPipelineCreateInfo &ci);

  uint64_t computePipelineHash(const ComputePipelineCreateInfo &ci);
  bool computePipelineEqual(const ComputePipelineCreateInfo &a,
                            const ComputePipelineCreateInfo &b);
  ComputePipeline *getComputePipeline(const ComputePipelineCreateInfo &ci);
};

} // namespace Ifrit::Engine::GraphicsBackend::VulkanGraphics