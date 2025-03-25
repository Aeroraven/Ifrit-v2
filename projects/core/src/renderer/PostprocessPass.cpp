#include "ifrit/core/renderer/PostprocessPass.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/util/FileOps.h"

using namespace Ifrit::GraphicsBackend::Rhi;
namespace Ifrit::Core {
IFRIT_APIDECL PostprocessPass::GPUShader *
PostprocessPass::createShaderFromFile(const String &shaderPath, const String &entry,
                                      GraphicsBackend::Rhi::RhiShaderStage stage) {
  auto rhi = m_app->getRhiLayer();
  String shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/Postprocess/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(shaderPath, shaderCodeVec, entry, stage, RhiShaderSourceType::GLSLCode);
}

IFRIT_APIDECL PostprocessPass::DrawPass *PostprocessPass::setupRenderPipeline(RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  PipelineAttachmentConfigs paCfg;
  auto rtCfg = renderTargets->getFormat();
  paCfg.m_colorFormats = {rtCfg.m_colorFormats[0]};
  paCfg.m_depthFormat = rtCfg.m_depthFormat;
  rtCfg.m_colorFormats = paCfg.m_colorFormats;

  DrawPass *pass = nullptr;
  if (m_renderPipelines.find(paCfg) != m_renderPipelines.end()) {
    pass = m_renderPipelines[paCfg];
  } else {
    pass = rhi->createGraphicsPass();
    auto vsShader = createShaderFromFile("Postproc.Common.vert.glsl", "main", RhiShaderStage::Vertex);
    auto fsShader = createShaderFromFile(m_cfg.fragPath, "main", RhiShaderStage::Fragment);
    pass->setPixelShader(fsShader);
    pass->setVertexShader(vsShader);
    pass->setNumBindlessDescriptorSets(m_cfg.numDescriptorSets);
    pass->setPushConstSize(sizeof(u32) * m_cfg.numPushConstants);
    pass->setRenderTargetFormat(rtCfg);
    m_renderPipelines[paCfg] = pass;
  }
  return pass;
}

IFRIT_APIDECL PostprocessPass::ComputePass *PostprocessPass::setupComputePipeline() {
  auto rhi = m_app->getRhiLayer();
  if (m_computePipeline == nullptr) {
    m_computePipeline = rhi->createComputePass();
    auto csShader = createShaderFromFile(m_cfg.fragPath, "main", RhiShaderStage::Compute);
    m_computePipeline->setNumBindlessDescriptorSets(m_cfg.numDescriptorSets);
    m_computePipeline->setComputeShader(csShader);
    m_computePipeline->setPushConstSize(sizeof(u32) * m_cfg.numPushConstants);
  }
  return m_computePipeline;
}

IFRIT_APIDECL void PostprocessPass::renderInternal(PerFrameData *perframeData, RenderTargets *renderTargets,
                                                   const GPUCmdBuffer *cmd, const void *pushConstants,
                                                   const std::vector<GPUBindlessRef *> &bindDescs,
                                                   const String &scopeName) {
  auto pass = setupRenderPipeline(renderTargets);
  auto rhi = m_app->getRhiLayer();
  pass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    for (auto i = 0; i < bindDescs.size(); i++) {
      ctx->m_cmd->attachBindlessReferenceGraphics(pass, i + 1, bindDescs[i]);
    }
    ctx->m_cmd->setPushConst(pass, 0, m_cfg.numPushConstants * sizeof(u32), pushConstants);
    ctx->m_cmd->attachVertexBufferView(*rhi->getFullScreenQuadVertexBufferView());
    ctx->m_cmd->attachVertexBuffers(0, {rhi->getFullScreenQuadVertexBuffer().get()});
    ctx->m_cmd->drawInstanced(3, 1, 0, 0);
  });
  if (scopeName.size() > 0)
    cmd->beginScope(scopeName);
  pass->run(cmd, renderTargets, 0);
  if (scopeName.size() > 0)
    cmd->endScope();
}

IFRIT_APIDECL PostprocessPass::PostprocessPass(IApplication *app, const PostprocessPassConfig &cfg)
    : m_app(app), m_cfg(cfg) {

  if (cfg.isComputeShader) {
    iInfo("Creating compute shader pipeline");
    setupComputePipeline();
  }
}

} // namespace Ifrit::Core