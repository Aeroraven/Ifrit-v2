#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/common/util/FileOps.h"

using namespace Ifrit::GraphicsBackend::Rhi;

namespace Ifrit::Core {

IFRIT_APIDECL SyaroRenderer::GPUShader *SyaroRenderer::createShaderFromFile(
    const std::string &shaderPath, const std::string &entry,
    GraphicsBackend::Rhi::RhiShaderStage stage) {
  auto rhi = m_app->getRhiLayer();
  std::string shaderBasePath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  auto path = shaderBasePath + "/" + shaderPath;
  auto shaderCode = Ifrit::Common::Utility::readTextFile(path);
  std::vector<char> shaderCodeVec(shaderCode.begin(), shaderCode.end());
  return rhi->createShader(shaderCodeVec, entry, stage,
                           RhiShaderSourceType::GLSLCode);
}
IFRIT_APIDECL void SyaroRenderer::setupTextureShowPass() {
  auto rhi = m_app->getRhiLayer();

  auto vsShader = createShaderFromFile("Syaro.FullScreenCopy.vert.glsl", "main",
                                       RhiShaderStage::Vertex);
  auto fsShader = createShaderFromFile("Syaro.FullScreenCopy.frag.glsl", "main",
                                       RhiShaderStage::Fragment);

  m_textureShowPass = rhi->createGraphicsPass();
  m_textureShowPass->setVertexShader(vsShader);
  m_textureShowPass->setPixelShader(fsShader);
  m_textureShowPass->setNumBindlessDescriptorSets(1); // Only one texture

  Ifrit::GraphicsBackend::Rhi::RhiRenderTargetsFormat rtFmt;
  rtFmt.m_colorFormats = {RhiImageFormat::RHI_FORMAT_B8G8R8A8_SRGB};
  rtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
  m_textureShowPass->setRenderTargetFormat(rtFmt);
}

IFRIT_APIDECL void SyaroRenderer::setupVisibilityPass() {
  auto rhi = m_app->getRhiLayer();
  auto msShader = createShaderFromFile("Syaro.VisBuffer.mesh.glsl", "main",
                                       RhiShaderStage::Mesh);
  auto fsShader = createShaderFromFile("Syaro.VisBuffer.frag.glsl", "main",
                                       RhiShaderStage::Fragment);

  m_visibilityPass = rhi->createGraphicsPass();
  m_visibilityPass->setMeshShader(msShader);
  m_visibilityPass->setPixelShader(fsShader);
  m_visibilityPass->setNumBindlessDescriptorSets(2);

  Ifrit::GraphicsBackend::Rhi::RhiRenderTargetsFormat rtFmt;
  rtFmt.m_colorFormats = {RhiImageFormat::RHI_FORMAT_R32_UINT};
  rtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
  m_visibilityPass->setRenderTargetFormat(rtFmt);
}

IFRIT_APIDECL void SyaroRenderer::setupPersistentCullingPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.PersistentCulling.comp.glsl",
                                     "main", RhiShaderStage::Compute);

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
  buildPipelines(perframeData, GraphicsShaderPassType::Opaque,
                 perframeData.m_visRTs.get());
  prepareDeviceResources(perframeData);
  gatherAllInstances(perframeData);

  // Then draw
  auto rhi = m_app->getRhiLayer();
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  // set record function for each class
  bool isFirst = true;
  auto rtFormats = perframeData.m_visRTs->getFormat();
  PipelineAttachmentConfigs paFormat = {rtFormats.m_depthFormat,
                                        rtFormats.m_colorFormats};
  // TODO: Instance culling is material-agonistic, no separate dispatch required
  m_textureShowPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceGraphics(
        m_textureShowPass, 1, perframeData.m_visShowCombinedRef);
    ctx->m_cmd->attachVertexBufferView(
        *rhi->getFullScreenQuadVertexBufferView());
    ctx->m_cmd->attachVertexBuffers(
        0, {rhi->getFullScreenQuadVertexBuffer().get()});
    ctx->m_cmd->drawInstanced(3, 1, 0, 0);
  });
  m_cullingPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    // bind view buffer
    ctx->m_cmd->attachBindlessReferenceCompute(m_cullingPass, 1,
                                               perframeData.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceCompute(
        m_cullingPass, 2, perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
    ctx->m_cmd->attachBindlessReferenceCompute(m_cullingPass, 3,
                                               m_cullingDescriptor);
    auto numObjs = perframeData.m_allInstanceData.m_objectData.size();
    ctx->m_cmd->dispatch(numObjs, 1, 1);
  });
  m_visibilityPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    // bind view buffer
    ctx->m_cmd->attachBindlessReferenceGraphics(m_visibilityPass, 1,
                                                perframeData.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceGraphics(
        m_visibilityPass, 2, perframeData.m_allInstanceData.m_batchedObjBufRef);
    ctx->m_cmd->drawMeshTasksIndirect(m_indirectDrawBuffer, 0, 1, 0);
  });
  auto compTask = compq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) { m_cullingPass->run(cmd, 0); },
      cmdToWait, {});
  auto drawTask = drawq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        m_visibilityPass->run(cmd, perframeData.m_visRTs.get(), 0);
      },
      {compTask.get()}, {});
  auto showTask = drawq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        m_textureShowPass->run(cmd, renderTargets, 0);
      },
      {drawTask.get()}, {});
  return showTask;
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

IFRIT_APIDECL void
SyaroRenderer::gatherAllInstances(PerFrameData &perframeData) {
  uint32_t totalInstances = 0;
  for (auto x : perframeData.m_enabledEffects) {
    auto &effect = perframeData.m_shaderEffectData[x];
    totalInstances += effect.m_objectData.size();
  }
  auto rhi = m_app->getRhiLayer();
  if (perframeData.m_allInstanceData.m_lastObjectCount != totalInstances) {
    perframeData.m_allInstanceData.m_lastObjectCount = totalInstances;
    perframeData.m_allInstanceData.m_batchedObjectData =
        rhi->createStorageBufferShared(totalInstances * sizeof(PerObjectData),
                                       true, 0);
    perframeData.m_allInstanceData.m_batchedObjBufRef =
        rhi->createBindlessDescriptorRef();
    auto buf = perframeData.m_allInstanceData.m_batchedObjectData;
    perframeData.m_allInstanceData.m_batchedObjBufRef->addStorageBuffer(buf, 0);
  }
  perframeData.m_allInstanceData.m_objectData.resize(totalInstances);
  for (auto i = 0; auto &x : perframeData.m_enabledEffects) {
    auto &effect = perframeData.m_shaderEffectData[x];
    for (auto &obj : effect.m_objectData) {
      perframeData.m_allInstanceData.m_objectData[i] = obj;
      i++;
    }
  }
  auto activeBuf =
      perframeData.m_allInstanceData.m_batchedObjectData->getActiveBuffer();
  activeBuf->map();
  activeBuf->writeBuffer(perframeData.m_allInstanceData.m_objectData.data(),
                         perframeData.m_allInstanceData.m_objectData.size() *
                             sizeof(PerObjectData),
                         0);
  activeBuf->flush();
  activeBuf->unmap();
}

} // namespace Ifrit::Core