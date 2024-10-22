#pragma once
#include <memory>
#include <vkrenderer/include/engine/vkrenderer/EngineContext.h>
#include <vkrenderer/include/engine/vkrenderer/Shader.h>

namespace Ifrit::Engine::VkRenderer {

enum class CullMode { None, Front, Back };
enum class RasterizerTopology { TriangleList, Line };
struct GraphicsPipelineCreateInfo {
  uint32_t viewportCount;
  uint32_t scissorCount;
  RasterizerTopology topology;
  std::vector<VkFormat> colorAttachmentFormats;
  VkFormat depthAttachmentFormat;
  VkFormat stencilAttachmentFormat;
  std::vector<ShaderModule *> shaderModules;
  std::vector<VkDescriptorSetLayout> descriptorSetLayouts;
};
struct ComputePipelineCreateInfo {};

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
} // namespace Ifrit::Engine::VkRenderer