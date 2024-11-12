#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/common/util/FileOps.h"

using namespace Ifrit::GraphicsBackend::Rhi;

namespace Ifrit::Core {
IFRIT_APIDECL void SyaroRenderer::setupCullingPass() {
  auto rhi = m_app->getRhiLayer();
  std::string shaderPath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  shaderPath += "/Syaro.LodCulling.comp.glsl";
  auto shaderCode = Ifrit::Common::Utility::readTextFile(shaderPath);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  auto shader =
      rhi->createShader(shaderCodeVec, "main", RhiShaderStage::Compute,
                        RhiShaderSourceType::GLSLCode);

  m_cullingPass = rhi->createComputePass();
  m_cullingPass->setComputeShader(shader);
  m_cullingPass->setNumBindlessDescriptorSets(3);

  m_indirectDrawBuffer = rhi->createIndirectMeshDrawBufferDevice(1);
  m_indirectDrawBufferId = rhi->registerStorageBuffer(m_indirectDrawBuffer);
  m_cullingDescriptor = rhi->createBindlessDescriptorRef();
  m_cullingDescriptor->addStorageBuffer(m_indirectDrawBuffer, 0);
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::render(
    PerFrameData &perframeData, SyaroRenderer::RenderTargets *renderTargets,
    const std::vector<SyaroRenderer::GPUCommandSubmission *> &cmdToWait) {

  // Some setups
  visbilityBufferSetup(perframeData, renderTargets);
  buildPipelines(perframeData, GraphicsShaderPassType::Opaque, perframeData.m_visRTs.get());
  prepareDeviceResources(perframeData);
  

  // Then draw
  auto rhi = m_app->getRhiLayer();
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  // set record function for each class
  bool isFirst = true;
  std::unique_ptr<GPUCommandSubmission> drawTask;
  auto rtFormats = perframeData.m_visRTs->getFormat();
  PipelineAttachmentConfigs paFormat = {rtFormats.m_depthFormat,
                                        rtFormats.m_colorFormats};
  // TODO: Instance culling is material-agonistic, no separate dispatch required
  for (auto &shaderEffect : perframeData.m_shaderEffectData) {
    auto pass = shaderEffect.m_materials[0]
                    ->m_effectTemplates[GraphicsShaderPassType::Opaque]
                    .m_drawPasses[paFormat];
    m_textureShowPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
      ctx->m_cmd->attachBindlessReferenceGraphics(
          m_textureShowPass, 1, perframeData.m_visShowCombinedRef);
      ctx->m_cmd->attachVertexBufferView(
          *rhi->getFullScreenQuadVertexBufferView());
      ctx->m_cmd->attachVertexBuffers(
          0, {rhi->getFullScreenQuadVertexBuffer().get()});
      ctx->m_cmd->drawInstanced(3, 1, 0, 0);
    });
    pass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
      // bind view buffer
      ctx->m_cmd->attachBindlessReferenceGraphics(
          pass, 1, perframeData.m_viewBindlessRef);
      auto ref = shaderEffect.m_materials[0]
                     ->m_effectTemplates[GraphicsShaderPassType::Opaque];
      ctx->m_cmd->attachBindlessReferenceGraphics(
          pass, 2, shaderEffect.m_batchedObjBufRef);
      ctx->m_cmd->drawMeshTasksIndirect(m_indirectDrawBuffer, 0, 1, 0);
    });
    if (m_cullingPass == nullptr) {
      setupCullingPass();
    }
    m_cullingPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
      // bind view buffer
      ctx->m_cmd->attachBindlessReferenceCompute(
          m_cullingPass, 1, perframeData.m_viewBindlessRef);
      auto ref = shaderEffect.m_materials[0]
                     ->m_effectTemplates[GraphicsShaderPassType::Opaque];
      ctx->m_cmd->attachBindlessReferenceCompute(
          m_cullingPass, 2, shaderEffect.m_batchedObjBufRef);
      ctx->m_cmd->attachBindlessReferenceCompute(m_cullingPass, 3,
                                                 m_cullingDescriptor);
      auto numObjs = shaderEffect.m_materials.size();
      ctx->m_cmd->dispatch(numObjs, 1, 1);
    });
     
    // Visibility buffer
    auto semaToWait =
        (isFirst) ? cmdToWait
                  : std::vector<SyaroRenderer::GPUCommandSubmission *>();
    isFirst = false;
    if (drawTask != nullptr) {
      semaToWait.push_back(drawTask.get());
    }
    auto compTask = compq->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) { m_cullingPass->run(cmd, 0); },
        semaToWait, {});
    drawTask = drawq->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) {
          pass->run(cmd, perframeData.m_visRTs.get(), 0);
        },
        {compTask.get()}, {});
  }
  auto showTask = drawq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        m_textureShowPass->run(cmd, renderTargets, 0);
      },
      {drawTask.get()}, {});
  return showTask;
}

IFRIT_APIDECL void SyaroRenderer::setupTextureShowPass() {
  auto rhi = m_app->getRhiLayer();

  std::string vsShaderPath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  std::string fsShaderPath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  vsShaderPath += "/Syaro.FullScreenCopy.vert.glsl";
  fsShaderPath += "/Syaro.FullScreenCopy.frag.glsl";

  auto vsShaderCode = Ifrit::Common::Utility::readTextFile(vsShaderPath);
  auto fsShaderCode = Ifrit::Common::Utility::readTextFile(fsShaderPath);
  std::vector<char> vsShaderCodeVec(vsShaderCode.begin(), vsShaderCode.end());
  auto vsShader =
      rhi->createShader(vsShaderCodeVec, "main", RhiShaderStage::Vertex,
                        RhiShaderSourceType::GLSLCode);
  std::vector<char> fsShaderCodeVec(fsShaderCode.begin(), fsShaderCode.end());
  auto fsShader =
      rhi->createShader(fsShaderCodeVec, "main", RhiShaderStage::Fragment,
                        RhiShaderSourceType::GLSLCode);

  m_textureShowPass = rhi->createGraphicsPass();
  m_textureShowPass->setVertexShader(vsShader);
  m_textureShowPass->setPixelShader(fsShader);
  m_textureShowPass->setNumBindlessDescriptorSets(1); // Only one texture

  Ifrit::GraphicsBackend::Rhi::RhiRenderTargetsFormat rtFmt;
  rtFmt.m_colorFormats = {RhiImageFormat::RHI_FORMAT_B8G8R8A8_SRGB};
  rtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
  m_textureShowPass->setRenderTargetFormat(rtFmt);
}
IFRIT_APIDECL void
SyaroRenderer::visbilityBufferSetup(PerFrameData &perframeData,
                                    RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  auto rtArea = renderTargets->getRenderArea();
  bool createCond = (perframeData.m_visibilityBuffer == nullptr);
  if (!createCond) {
    auto visHeight = perframeData.m_visibilityBuffer->getHeight();
    auto visWidth = perframeData.m_visibilityBuffer->getWidth();
    auto rtSize = renderTargets->getRenderArea();
    createCond = (visHeight != rtSize.height + rtSize.x ||
                  visWidth != rtSize.width + rtSize.y);
  }
  if (!createCond) {
    return;
  }
  // It seems nanite's paper uses R32G32 for mesh visibility,
  // but I wonder the depth is implicitly calculated from the depth buffer
  // so here I use R32 for visibility buffer
  auto visBuffer = rhi->createRenderTargetTexture(
      rtArea.width + rtArea.x, rtArea.height + rtArea.y,
      PerFrameData::c_visibilityFormat);
  auto visDepth = rhi->createDepthRenderTexture(rtArea.width + rtArea.x,
                                                rtArea.height + rtArea.y);
  perframeData.m_visibilityBuffer = visBuffer;
  perframeData.m_visPassDepth = visDepth;
  perframeData.m_visColorRT = rhi->createRenderTarget(
      visBuffer.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Clear);
  perframeData.m_visDepthRT = rhi->createRenderTargetDepthStencil(
      visDepth, {{}, 1.0f}, RhiRenderTargetLoadOp::Clear);

  perframeData.m_visRTs = rhi->createRenderTargets();
  perframeData.m_visRTs->setColorAttachments({perframeData.m_visColorRT.get()});
  perframeData.m_visRTs->setDepthStencilAttachment(
      perframeData.m_visDepthRT.get());
  perframeData.m_visRTs->setRenderArea(renderTargets->getRenderArea());

  // Then a sampler
  perframeData.m_visibilitySampler = rhi->createTrivialSampler();
  perframeData.m_visShowCombinedRef = rhi->createBindlessDescriptorRef();
  perframeData.m_visShowCombinedRef->addCombinedImageSampler(
      perframeData.m_visibilityBuffer.get(),
      perframeData.m_visibilitySampler.get(), 0);

  // Command  buffer recording
  if (m_textureShowPass == nullptr) {
    setupTextureShowPass();
  }
  
}

} // namespace Ifrit::Core