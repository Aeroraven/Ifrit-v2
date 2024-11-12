#include "ifrit/core/renderer/SimpleRenderer.h"
#include "ifrit/common/util/FileOps.h"

using namespace Ifrit::GraphicsBackend::Rhi;

namespace Ifrit::Core {
IFRIT_APIDECL void SimpleRenderer::setupCullingPass() {
  auto rhi = m_app->getRhiLayer();
  std::string shaderPath = IFRIT_CORELIB_SHARED_SHADER_PATH;
  shaderPath += "/LodCulling.comp.glsl";
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

IFRIT_APIDECL std::unique_ptr<SimpleRenderer::GPUCommandSubmission>
SimpleRenderer::render(
    PerFrameData &perframeData, SimpleRenderer::RenderTargets *renderTargets,
    const std::vector<SimpleRenderer::GPUCommandSubmission *> &cmdToWait) {

  buildPipelines(perframeData, GraphicsShaderPassType::Opaque, renderTargets);
  prepareDeviceResources(perframeData);

  auto rhi = m_app->getRhiLayer();
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  // set record function for each class
  bool isFirst = true;
  std::unique_ptr<GPUCommandSubmission> drawTask;
  auto rtFormats = renderTargets->getFormat();
  PipelineAttachmentConfigs paFormat = {rtFormats.m_depthFormat,
                                        rtFormats.m_colorFormats};
  for (auto &shaderEffect : perframeData.m_shaderEffectData) {
    auto pass = shaderEffect.m_materials[0]
                    ->m_effectTemplates[GraphicsShaderPassType::Opaque]
                    .m_drawPasses[paFormat];
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

    // record draw command
    auto semaToWait =
        (isFirst) ? cmdToWait
                  : std::vector<SimpleRenderer::GPUCommandSubmission *>();
    isFirst = false;
    if (drawTask != nullptr) {
      semaToWait.push_back(drawTask.get());
    }
    auto compTask = compq->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) { m_cullingPass->run(cmd, 0); },
        semaToWait, {});
    drawTask = drawq->runAsyncCommand(
        [&](const RhiCommandBuffer *cmd) { pass->run(cmd, renderTargets, 0); },
        {compTask.get()}, {});
  }
  return drawTask;
}
} // namespace Ifrit::Core