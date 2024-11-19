#include "ifrit/core/renderer/SyaroRenderer.h"
#include "ifrit/common/util/FileOps.h"

using namespace Ifrit::GraphicsBackend::Rhi;

namespace Ifrit::Core {

struct GPUHiZDesc {
  uint32_t m_width;
  uint32_t m_height;
};

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
  m_visibilityPass->setNumBindlessDescriptorSets(3);
  m_visibilityPass->setPushConstSize(sizeof(uint32_t));

  Ifrit::GraphicsBackend::Rhi::RhiRenderTargetsFormat rtFmt;
  rtFmt.m_colorFormats = {RhiImageFormat::RHI_FORMAT_R32_UINT};
  rtFmt.m_depthFormat = RhiImageFormat::RHI_FORMAT_D32_SFLOAT;
  m_visibilityPass->setRenderTargetFormat(rtFmt);
}

IFRIT_APIDECL void SyaroRenderer::setupInstanceCullingPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.InstanceCulling.comp.glsl", "main",
                                     RhiShaderStage::Compute);

  m_instanceCullingPass = rhi->createComputePass();
  m_instanceCullingPass->setComputeShader(shader);
  m_instanceCullingPass->setNumBindlessDescriptorSets(4);
  m_instanceCullingPass->setPushConstSize(sizeof(uint32_t));
}

IFRIT_APIDECL void SyaroRenderer::setupPersistentCullingPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.PersistentCulling.comp.glsl",
                                     "main", RhiShaderStage::Compute);

  m_persistentCullingPass = rhi->createComputePass();
  m_persistentCullingPass->setComputeShader(shader);
  m_persistentCullingPass->setNumBindlessDescriptorSets(5);
  m_persistentCullingPass->setPushConstSize(sizeof(uint32_t));

  m_indirectDrawBuffer = rhi->createIndirectMeshDrawBufferDevice(
      1, Ifrit::GraphicsBackend::Rhi::RhiBufferUsage::
             RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  m_indirectDrawBufferId = rhi->registerStorageBuffer(m_indirectDrawBuffer);
  m_persistCullDesc = rhi->createBindlessDescriptorRef();
  m_persistCullDesc->addStorageBuffer(m_indirectDrawBuffer, 0);
}

IFRIT_APIDECL void SyaroRenderer::setupHiZPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.HiZ.comp.glsl", "main",
                                     RhiShaderStage::Compute);

  m_hizPass = rhi->createComputePass();
  m_hizPass->setComputeShader(shader);
  m_hizPass->setNumBindlessDescriptorSets(1);
}

IFRIT_APIDECL void SyaroRenderer::setupEmitDepthTargetsPass() {
  auto rhi = m_app->getRhiLayer();
  auto shader = createShaderFromFile("Syaro.EmitDepthTarget.comp.glsl", "main",
                                     RhiShaderStage::Compute);

  m_emitDepthTargetsPass = rhi->createComputePass();
  m_emitDepthTargetsPass->setComputeShader(shader);
  m_emitDepthTargetsPass->setNumBindlessDescriptorSets(4);
  m_emitDepthTargetsPass->setPushConstSize(2 * sizeof(uint32_t));
}

IFRIT_APIDECL void SyaroRenderer::setupMaterialClassifyPass() {
  auto rhi = m_app->getRhiLayer();
  // Count pass
  if constexpr (true) {
    auto shader = createShaderFromFile("Syaro.ClassifyMaterial.Count.comp.glsl",
                                       "main", RhiShaderStage::Compute);
    m_matclassCountPass = rhi->createComputePass();
    m_matclassCountPass->setComputeShader(shader);
    m_matclassCountPass->setNumBindlessDescriptorSets(1);
    m_matclassCountPass->setPushConstSize(sizeof(uint32_t) * 3);
  }
  // Reserve pass
  if constexpr (true) {
    auto shader =
        createShaderFromFile("Syaro.ClassifyMaterial.Reserve.comp.glsl", "main",
                             RhiShaderStage::Compute);
    m_matclassReservePass = rhi->createComputePass();
    m_matclassReservePass->setComputeShader(shader);
    m_matclassReservePass->setNumBindlessDescriptorSets(1);
    m_matclassReservePass->setPushConstSize(sizeof(uint32_t) * 3);
  }
  // Scatter pass
  if constexpr (true) {
    auto shader =
        createShaderFromFile("Syaro.ClassifyMaterial.Scatter.comp.glsl", "main",
                             RhiShaderStage::Compute);
    m_matclassScatterPass = rhi->createComputePass();
    m_matclassScatterPass->setComputeShader(shader);
    m_matclassScatterPass->setNumBindlessDescriptorSets(1);
    m_matclassScatterPass->setPushConstSize(sizeof(uint32_t) * 3);
  }
  // Debug pass
  if constexpr (true) {
    auto shader = createShaderFromFile("Syaro.ClassifyMaterial.Debug.comp.glsl",
                                       "main", RhiShaderStage::Compute);
    m_matclassDebugPass = rhi->createComputePass();
    m_matclassDebugPass->setComputeShader(shader);
    m_matclassDebugPass->setNumBindlessDescriptorSets(1);
    m_matclassDebugPass->setPushConstSize(sizeof(uint32_t) * 3);
  }
}

IFRIT_APIDECL void
SyaroRenderer::materialClassifyBufferSetup(PerFrameData &perframeData,
                                           RenderTargets *renderTargets) {
  auto numMaterials = perframeData.m_enabledEffects.size();
  auto rhi = m_app->getRhiLayer();
  auto renderArea = renderTargets->getRenderArea();
  auto width = renderArea.width + renderArea.x;
  auto height = renderArea.height + renderArea.y;
  auto totalSize = width * height;
  bool needRecreate = false;
  bool needRecreateMat = false;
  bool needRecreatePixel = false;
  if (perframeData.m_matClassSupportedNumMaterials < numMaterials ||
      perframeData.m_matClassCountBuffer == nullptr) {
    needRecreate = true;
    needRecreateMat = true;
  }
  if (perframeData.m_matClassSupportedNumPixels < totalSize) {
    needRecreate = true;
    needRecreatePixel = true;
  }
  if (!needRecreate) {
    return;
  }
  if (needRecreateMat) {
    perframeData.m_matClassSupportedNumMaterials = numMaterials;
    auto createSize = cMatClassCounterBufferSizeBase +
                      cMatClassCounterBufferSizeMult * numMaterials;
    perframeData.m_matClassCountBuffer = rhi->createStorageBufferDevice(
        createSize, RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

    perframeData.m_matClassIndirectDispatchBuffer =
        rhi->createStorageBufferDevice(
            sizeof(uint32_t) * 3,
            RhiBufferUsage::RHI_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  }
  if (needRecreatePixel) {
    perframeData.m_matClassSupportedNumPixels = totalSize;
    perframeData.m_matClassFinalBuffer = rhi->createStorageBufferDevice(
        totalSize * sizeof(uint32_t),
        RhiBufferUsage::RHI_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    perframeData.m_matClassPixelOffsetBuffer = rhi->createStorageBufferDevice(
        totalSize * sizeof(uint32_t),
        RhiBufferUsage::RHI_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    perframeData.m_matClassDebug = rhi->createRenderTargetTexture(
        width, height, RhiImageFormat::RHI_FORMAT_R32_UINT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT);
    perframeData.m_matClassDebug = rhi->createRenderTargetTexture(
        width, height, RhiImageFormat::RHI_FORMAT_R32_UINT,
        RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT |
            RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  if (needRecreate) {
    perframeData.m_matClassDesc = rhi->createBindlessDescriptorRef();
    perframeData.m_matClassDesc->addUAVImage(
        perframeData.m_velocityMaterial.get(), {0, 0, 1, 1}, 0);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassCountBuffer, 1);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassFinalBuffer, 2);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassPixelOffsetBuffer, 3);
    perframeData.m_matClassDesc->addStorageBuffer(
        perframeData.m_matClassIndirectDispatchBuffer, 4);
    perframeData.m_matClassDesc->addUAVImage(perframeData.m_matClassDebug.get(),
                                             {0, 0, 1, 1}, 5);

    perframeData.m_matClassBarrier.clear();
    RhiResourceBarrier barrierCountBuffer;
    barrierCountBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierCountBuffer.m_uav.m_buffer = perframeData.m_matClassCountBuffer;
    barrierCountBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierFinalBuffer;
    barrierFinalBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierFinalBuffer.m_uav.m_buffer = perframeData.m_matClassFinalBuffer;
    barrierFinalBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierPixelOffsetBuffer;
    barrierPixelOffsetBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierPixelOffsetBuffer.m_uav.m_buffer =
        perframeData.m_matClassPixelOffsetBuffer;
    barrierPixelOffsetBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierIndirectDispatchBuffer;
    barrierIndirectDispatchBuffer.m_type = RhiBarrierType::UAVAccess;
    barrierIndirectDispatchBuffer.m_uav.m_buffer =
        perframeData.m_matClassIndirectDispatchBuffer;
    barrierIndirectDispatchBuffer.m_uav.m_type = RhiResourceType::Buffer;

    RhiResourceBarrier barrierDebug;
    barrierDebug.m_type = RhiBarrierType::UAVAccess;
    barrierDebug.m_uav.m_texture = perframeData.m_matClassDebug.get();
    barrierDebug.m_uav.m_type = RhiResourceType::Texture;

    perframeData.m_matClassBarrier.push_back(barrierCountBuffer);
    perframeData.m_matClassBarrier.push_back(barrierFinalBuffer);
    perframeData.m_matClassBarrier.push_back(barrierPixelOffsetBuffer);
    perframeData.m_matClassBarrier.push_back(barrierIndirectDispatchBuffer);
    perframeData.m_matClassBarrier.push_back(barrierDebug);
  }
}

IFRIT_APIDECL void
SyaroRenderer::depthTargetsSetup(PerFrameData &perframeData,
                                 RenderTargets *renderTargets) {
  auto rhi = m_app->getRhiLayer();
  auto rtArea = renderTargets->getRenderArea();
  if (perframeData.m_velocityMaterial != nullptr)
    return;
  perframeData.m_velocityMaterial = rhi->createRenderTargetTexture(
      rtArea.width + rtArea.x, rtArea.height + rtArea.y,
      RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT,
      RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
  perframeData.m_velocityMaterialDesc = rhi->createBindlessDescriptorRef();
  perframeData.m_velocityMaterialDesc->addUAVImage(
      perframeData.m_velocityMaterial.get(), {0, 0, 1, 1}, 0);
  perframeData.m_velocityMaterialDesc->addCombinedImageSampler(
      perframeData.m_visibilityBuffer.get(),
      perframeData.m_visibilitySampler.get(), 1);
}

IFRIT_APIDECL void
SyaroRenderer::recreateInstanceCullingBuffers(uint32_t newMaxInstances) {
  if (m_maxSupportedInstances == 0 ||
      m_maxSupportedInstances < newMaxInstances) {
    auto rhi = m_app->getRhiLayer();
    m_maxSupportedInstances = newMaxInstances;
    m_instCullDiscardObj =
        rhi->createStorageBufferDevice(newMaxInstances * sizeof(uint32_t), 0);
    m_instCullPassedObj =
        rhi->createStorageBufferDevice(newMaxInstances * sizeof(uint32_t), 0);
    m_persistCullIndirectDispatch = rhi->createStorageBufferDevice(
        sizeof(uint32_t) * 9, Ifrit::GraphicsBackend::Rhi::RhiBufferUsage::
                                      RHI_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                  Ifrit::GraphicsBackend::Rhi::RhiBufferUsage::
                                      RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_instCullDesc = rhi->createBindlessDescriptorRef();
    m_instCullDesc->addStorageBuffer(m_instCullDiscardObj, 0);
    m_instCullDesc->addStorageBuffer(m_instCullPassedObj, 1);
    m_instCullDesc->addStorageBuffer(m_persistCullIndirectDispatch, 2);
  }
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::renderEmitDepthTargets(
    PerFrameData &perframeData, SyaroRenderer::RenderTargets *renderTargets,
    const std::vector<SyaroRenderer::GPUCommandSubmission *> &cmdToWait) {
  auto rhi = m_app->getRhiLayer();
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);

  m_emitDepthTargetsPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        ctx->m_cmd->imageBarrier(
            perframeData.m_velocityMaterial.get(), RhiResourceState::Undefined,
            RhiResourceState::UAVStorageImage, {0, 0, 1, 1});
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 1, perframeData.m_viewBindlessRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 2,
            perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 3, perframeData.m_allFilteredMeshletsDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_emitDepthTargetsPass, 4, perframeData.m_velocityMaterialDesc);
        uint32_t pcData[2] = {perframeData.m_viewData.m_renderWidth,
                              perframeData.m_viewData.m_renderHeight};
        ctx->m_cmd->setPushConst(m_emitDepthTargetsPass, 0,
                                 sizeof(uint32_t) * 2, &pcData[0]);

        uint32_t wgX =
            (pcData[0] + cEmitDepthGroupSizeX - 1) / cEmitDepthGroupSizeX;
        uint32_t wgY =
            (pcData[1] + cEmitDepthGroupSizeY - 1) / cEmitDepthGroupSizeY;
        ctx->m_cmd->dispatch(wgX, wgY, 1);
      });

  auto emitDepthTask = compq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        cmd->beginScope("Syaro: Emit Depth Targets");
        m_emitDepthTargetsPass->run(cmd, 0);
        cmd->endScope();
      },
      cmdToWait, {});
  return emitDepthTask;
}
IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::renderTwoPassOcclCulling(
    CullingPass cullPass, PerFrameData &perframeData,
    SyaroRenderer::RenderTargets *renderTargets,
    const std::vector<SyaroRenderer::GPUCommandSubmission *> &cmdToWait) {
  auto rhi = m_app->getRhiLayer();
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);

  int pcData[2] = {0, 1};
  m_instanceCullingPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        // TODO: this assumes the compute queue can TRANSFER
        if (cullPass == CullingPass::First) {
          ctx->m_cmd->uavBufferClear(m_persistCullIndirectDispatch, 0);
          ctx->m_cmd->uavBufferBarrier(m_persistCullIndirectDispatch);
        }

        ctx->m_cmd->attachBindlessReferenceCompute(
            m_instanceCullingPass, 1, perframeData.m_viewBindlessRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_instanceCullingPass, 2,
            perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
        ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 3,
                                                   m_instCullDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(m_instanceCullingPass, 4,
                                                   perframeData.m_hizTestDesc);
        auto numObjs = perframeData.m_allInstanceData.m_objectData.size();
        if (cullPass == CullingPass::First) {
          ctx->m_cmd->setPushConst(m_instanceCullingPass, 0, sizeof(uint32_t),
                                   &pcData[0]);
          ctx->m_cmd->dispatch(numObjs, 1, 1);
        } else if (cullPass == CullingPass::Second) {
          ctx->m_cmd->setPushConst(m_instanceCullingPass, 0, sizeof(uint32_t),
                                   &pcData[1]);
          ctx->m_cmd->dispatchIndirect(m_persistCullIndirectDispatch,
                                       3 * sizeof(uint32_t));
        }
      });

  m_persistentCullingPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        if (cullPass == CullingPass::First) {
          ctx->m_cmd->uavBufferClear(perframeData.m_allFilteredMeshletsCount,
                                     0);
          ctx->m_cmd->uavBufferBarrier(perframeData.m_allFilteredMeshletsCount);
          ctx->m_cmd->uavBufferClear(m_indirectDrawBuffer, 0);
          ctx->m_cmd->uavBufferBarrier(m_indirectDrawBuffer);
        }
        // bind view buffer
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_persistentCullingPass, 1, perframeData.m_viewBindlessRef);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_persistentCullingPass, 2,
            perframeData.m_shaderEffectData[0].m_batchedObjBufRef);
        ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 3,
                                                   m_persistCullDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(
            m_persistentCullingPass, 4, perframeData.m_allFilteredMeshletsDesc);
        ctx->m_cmd->attachBindlessReferenceCompute(m_persistentCullingPass, 5,
                                                   m_instCullDesc);
        if (cullPass == CullingPass::First) {
          ctx->m_cmd->setPushConst(m_persistentCullingPass, 0, sizeof(uint32_t),
                                   &pcData[0]);
          ctx->m_cmd->dispatchIndirect(m_persistCullIndirectDispatch, 0);
        } else if (cullPass == CullingPass::Second) {
          ctx->m_cmd->setPushConst(m_persistentCullingPass, 0, sizeof(uint32_t),
                                   &pcData[1]);
          ctx->m_cmd->dispatchIndirect(m_persistCullIndirectDispatch,
                                       6 * sizeof(uint32_t));
        }
      });
  m_visibilityPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    // bind view buffer

    ctx->m_cmd->attachBindlessReferenceGraphics(m_visibilityPass, 1,
                                                perframeData.m_viewBindlessRef);
    ctx->m_cmd->attachBindlessReferenceGraphics(
        m_visibilityPass, 2, perframeData.m_allInstanceData.m_batchedObjBufRef);
    ctx->m_cmd->attachBindlessReferenceGraphics(
        m_visibilityPass, 3, perframeData.m_allFilteredMeshletsDesc);
    if (cullPass == CullingPass::First) {
      ctx->m_cmd->setPushConst(m_visibilityPass, 0, sizeof(uint32_t),
                               &pcData[0]);
      ctx->m_cmd->drawMeshTasksIndirect(perframeData.m_allFilteredMeshletsCount,
                                        0, 1, 0);
    } else {
      ctx->m_cmd->setPushConst(m_visibilityPass, 0, sizeof(uint32_t),
                               &pcData[1]);
      ctx->m_cmd->drawMeshTasksIndirect(perframeData.m_allFilteredMeshletsCount,
                                        sizeof(uint32_t) * 3, 1, 0);
    }
  });

  m_hizPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    auto rtHeight = perframeData.m_hizTexture->getHeight();
    auto rtWidth = perframeData.m_hizTexture->getWidth();
    Ifrit::GraphicsBackend::Rhi::RhiImageSubResource subRes0 = {0, 0, 1, 1};
    ctx->m_cmd->imageBarrier(perframeData.m_hizTexture.get(),
                             RhiResourceState::Undefined,
                             RhiResourceState::UAVStorageImage, subRes0);
    for (int i = 0; i < perframeData.m_hizIter; i++) {
      auto desc = perframeData.m_hizDescs[i];
      ctx->m_cmd->attachBindlessReferenceCompute(m_hizPass, 1, desc);
      auto wgX = (rtWidth + cHiZGroupSizeX - 1) / cHiZGroupSizeX;
      auto wgY = (rtHeight + cHiZGroupSizeY - 1) / cHiZGroupSizeY;
      ctx->m_cmd->dispatch(wgX, wgY, 1);
      rtWidth = std::max(1u, rtWidth / 2);
      rtHeight = std::max(1u, rtHeight / 2);
      if (i == perframeData.m_hizIter - 1) {
        return;
      }
      Ifrit::GraphicsBackend::Rhi::RhiImageSubResource subRes = {i, 0, 1, 1};
      ctx->m_cmd->imageBarrier(perframeData.m_hizTexture.get(),
                               RhiResourceState::UAVStorageImage,
                               RhiResourceState::UAVStorageImage, subRes);
      Ifrit::GraphicsBackend::Rhi::RhiImageSubResource subRes1 = {i + 1, 0, 1,
                                                                  1};
      ctx->m_cmd->imageBarrier(perframeData.m_hizTexture.get(),
                               RhiResourceState::Undefined,
                               RhiResourceState::UAVStorageImage, subRes1);
    }
  });

  auto instanceCullTask = compq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        cmd->beginScope("Syaro: Instance Culling Pass");
        m_instanceCullingPass->run(cmd, 0);
        cmd->endScope();
      },
      cmdToWait, {});
  auto persistCullTask = compq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        cmd->beginScope("Syaro: Persistent Culling Pass");
        m_persistentCullingPass->run(cmd, 0);
        cmd->endScope();
      },
      {instanceCullTask.get()}, {});
  auto drawTask = drawq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        if (cullPass == CullingPass::First) {
          cmd->beginScope("Syaro: Visibility Pass, First");
          m_visibilityPass->run(cmd, perframeData.m_visRTs.get(), 0);
          cmd->endScope();
        } else {
          // we wont clear the visibility buffer in the second pass
          cmd->beginScope("Syaro: Visibility Pass, Second");
          m_visibilityPass->run(cmd, perframeData.m_visRTs2.get(), 0);
          cmd->endScope();
        }
      },
      {persistCullTask.get()}, {});

  auto hizTask = compq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        cmd->beginScope("Syaro: HiZ Pass");
        m_hizPass->run(cmd, 0);
        cmd->endScope();
      },
      {drawTask.get()}, {});
  return hizTask;
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::renderMaterialClassify(
    PerFrameData &perframeData, RenderTargets *renderTargets,
    const std::vector<GPUCommandSubmission *> &cmdToWait) {
  auto rhi = m_app->getRhiLayer();
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto totalMaterials = perframeData.m_enabledEffects.size();

  auto renderArea = renderTargets->getRenderArea();
  auto width = renderArea.width + renderArea.x;
  auto height = renderArea.height + renderArea.y;
  uint32_t pcData[3] = {width, height, totalMaterials};

  constexpr uint32_t pTileWidth =
      cMatClassQuadSize * cMatClassGroupSizeCountScatterX;
  constexpr uint32_t pTileHeight =
      cMatClassQuadSize * cMatClassGroupSizeCountScatterY;

  // Counting
  auto wgX = (width + pTileWidth - 1) / pTileWidth;
  auto wgY = (height + pTileHeight - 1) / pTileHeight;
  m_matclassCountPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->uavBufferClear(perframeData.m_matClassCountBuffer, 0);
    ctx->m_cmd->uavBufferBarrier(perframeData.m_matClassCountBuffer);

    ctx->m_cmd->attachBindlessReferenceCompute(m_matclassCountPass, 1,
                                               perframeData.m_matClassDesc);
    ctx->m_cmd->setPushConst(m_matclassCountPass, 0, sizeof(uint32_t) * 3,
                             &pcData[0]);
    ctx->m_cmd->dispatch(wgX, wgY, 1);
  });

  // Reserving
  auto wgX2 = (totalMaterials + cMatClassGroupSizeReserveX - 1) /
              cMatClassGroupSizeReserveX;
  m_matclassReservePass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        ctx->m_cmd->attachBindlessReferenceCompute(m_matclassReservePass, 1,
                                                   perframeData.m_matClassDesc);
        ctx->m_cmd->setPushConst(m_matclassReservePass, 0, sizeof(uint32_t) * 3,
                                 &pcData[0]);
        ctx->m_cmd->dispatch(wgX2, 1, 1);
      });

  // Scatter
  m_matclassScatterPass->setRecordFunction(
      [&](const RhiRenderPassContext *ctx) {
        ctx->m_cmd->attachBindlessReferenceCompute(m_matclassScatterPass, 1,
                                                   perframeData.m_matClassDesc);
        ctx->m_cmd->setPushConst(m_matclassScatterPass, 0, sizeof(uint32_t) * 3,
                                 &pcData[0]);
        ctx->m_cmd->dispatch(wgX, wgY, 1);
      });

  // Debug
  m_matclassDebugPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->clearUAVImageFloat(perframeData.m_matClassDebug.get(),
                                   {0, 0, 1, 1}, {0.0f, 0.0f, 0.0f, 0.0f});
    ctx->m_cmd->imageBarrier(perframeData.m_matClassDebug.get(),
                             RhiResourceState::Common,
                             RhiResourceState::UAVStorageImage, {0, 0, 1, 1});
    ctx->m_cmd->attachBindlessReferenceCompute(m_matclassDebugPass, 1,
                                               perframeData.m_matClassDesc);
    ctx->m_cmd->setPushConst(m_matclassDebugPass, 0, sizeof(uint32_t) * 3,
                             &pcData[0]);
    ctx->m_cmd->dispatchIndirect(perframeData.m_matClassIndirectDispatchBuffer,
                                 0);
  });

  // Start rendering
  auto matclassTask = compq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        cmd->beginScope("Syaro: Material Classification");
        cmd->imageBarrier(perframeData.m_matClassDebug.get(),
                          RhiResourceState::Undefined, RhiResourceState::Common,
                          {0, 0, 1, 1});
        m_matclassCountPass->run(cmd, 0);
        cmd->resourceBarrier(perframeData.m_matClassBarrier);
        m_matclassReservePass->run(cmd, 0);
        cmd->resourceBarrier(perframeData.m_matClassBarrier);
        m_matclassScatterPass->run(cmd, 0);
        cmd->resourceBarrier(perframeData.m_matClassBarrier);
        m_matclassDebugPass->run(cmd, 0);
        cmd->endScope();
      },
      cmdToWait, {});

  return matclassTask;
}

IFRIT_APIDECL std::unique_ptr<SyaroRenderer::GPUCommandSubmission>
SyaroRenderer::render(
    PerFrameData &perframeData, SyaroRenderer::RenderTargets *renderTargets,
    const std::vector<SyaroRenderer::GPUCommandSubmission *> &cmdToWait) {
  // Some setups
  visibilityBufferSetup(perframeData, renderTargets);
  buildPipelines(perframeData, GraphicsShaderPassType::Opaque,
                 perframeData.m_visRTs.get());
  prepareDeviceResources(perframeData, renderTargets);
  gatherAllInstances(perframeData);
  recreateInstanceCullingBuffers(
      perframeData.m_allInstanceData.m_objectData.size());
  hizBufferSetup(perframeData, renderTargets);
  depthTargetsSetup(perframeData, renderTargets);
  materialClassifyBufferSetup(perframeData, renderTargets);

  // Then draw
  auto rhi = m_app->getRhiLayer();
  auto drawq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto compq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_COMPUTE_BIT);
  m_textureShowPass->setRecordFunction([&](const RhiRenderPassContext *ctx) {
    ctx->m_cmd->attachBindlessReferenceGraphics(
        m_textureShowPass, 1, perframeData.m_visShowCombinedRef);
    ctx->m_cmd->attachVertexBufferView(
        *rhi->getFullScreenQuadVertexBufferView());
    ctx->m_cmd->attachVertexBuffers(
        0, {rhi->getFullScreenQuadVertexBuffer().get()});
    ctx->m_cmd->drawInstanced(3, 1, 0, 0);
  });
  auto firstCullTask = renderTwoPassOcclCulling(
      CullingPass::First, perframeData, renderTargets, cmdToWait);
  auto secondCullTask = renderTwoPassOcclCulling(
      CullingPass::Second, perframeData, renderTargets, {firstCullTask.get()});
  auto emitDepthTask = renderEmitDepthTargets(perframeData, renderTargets,
                                              {secondCullTask.get()});
  auto matClassTask = renderMaterialClassify(perframeData, renderTargets,
                                             {emitDepthTask.get()});
  auto showTask = drawq->runAsyncCommand(
      [&](const RhiCommandBuffer *cmd) {
        cmd->beginScope("Syaro: Display Texture");
        m_textureShowPass->run(cmd, renderTargets, 0);
        cmd->endScope();
      },
      {matClassTask.get()}, {});
  return showTask;
}

IFRIT_APIDECL void SyaroRenderer::hizBufferSetup(PerFrameData &perframeData,
                                                 RenderTargets *renderTargets) {
  auto renderArea = renderTargets->getRenderArea();
  auto width = renderArea.width + renderArea.x;
  auto height = renderArea.height + renderArea.y;
  bool cond = (perframeData.m_hizTexture == nullptr);
  if (!cond && (perframeData.m_hizTexture->getWidth() != width ||
                perframeData.m_hizTexture->getHeight() != height)) {
    cond = true;
  }
  if (!cond) {
    return;
  }
  auto rhi = m_app->getRhiLayer();
  auto maxMip = int(std::floor(std::log2(std::max(width, height))) + 1);
  perframeData.m_hizTexture = rhi->createRenderTargetMipTexture(
      width, height, maxMip, RhiImageFormat::RHI_FORMAT_R32_SFLOAT,
      RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);

  uint32_t rWidth = renderArea.width + renderArea.x;
  uint32_t rHeight = renderArea.height + renderArea.y;
  perframeData.m_hizDepthSampler = rhi->createTrivialSampler();

  for (int i = 0; i < maxMip; i++) {
    auto desc = rhi->createBindlessDescriptorRef();
    auto hizTexSize = rhi->createStorageBufferDevice(
        sizeof(GPUHiZDesc), RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    GPUHiZDesc hizDesc = {rWidth, rHeight};
    auto stagedBuf = rhi->createStagedSingleBuffer(hizTexSize);
    auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
      stagedBuf->cmdCopyToDevice(cmd, &hizDesc, sizeof(GPUHiZDesc), 0);
    });
    rWidth = std::max(1u, rWidth / 2);
    rHeight = std::max(1u, rHeight / 2);
    desc->addCombinedImageSampler(perframeData.m_visPassDepth,
                                  perframeData.m_hizDepthSampler.get(), 0);
    desc->addUAVImage(perframeData.m_hizTexture.get(),
                      {static_cast<uint32_t>(std::max(0, i - 1)), 0, 1, 1}, 1);
    desc->addUAVImage(perframeData.m_hizTexture.get(),
                      {static_cast<uint32_t>(i), 0, 1, 1}, 2);
    desc->addStorageBuffer(hizTexSize, 3);
    perframeData.m_hizDescs.push_back(desc);
  }
  perframeData.m_hizIter = maxMip;

  // For hiz-testing, we need to create a descriptor for the hiz texture
  // Seems UAV/Storage image does not support mip-levels
  for (int i = 0; i < maxMip; i++) {
    perframeData.m_hizTestMips.push_back(rhi->registerUAVImage(
        perframeData.m_hizTexture.get(), {static_cast<uint32_t>(i), 0, 1, 1}));
    perframeData.m_hizTestMipsId.push_back(
        perframeData.m_hizTestMips.back()->getActiveId());
  }
  perframeData.m_hizTestMipsBuffer = rhi->createStorageBufferDevice(
      sizeof(uint32_t) * maxMip,
      RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
  auto staged = rhi->createStagedSingleBuffer(perframeData.m_hizTestMipsBuffer);
  tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
    staged->cmdCopyToDevice(cmd, perframeData.m_hizTestMipsId.data(),
                            sizeof(uint32_t) * maxMip, 0);
  });
  perframeData.m_hizTestDesc = rhi->createBindlessDescriptorRef();
  perframeData.m_hizTestDesc->addStorageBuffer(perframeData.m_hizTestMipsBuffer,
                                               0);
}

IFRIT_APIDECL void
SyaroRenderer::visibilityBufferSetup(PerFrameData &perframeData,
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
      PerFrameData::c_visibilityFormat,
      RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT);
  auto visDepth = rhi->createDepthRenderTexture(rtArea.width + rtArea.x,
                                                rtArea.height + rtArea.y);
  perframeData.m_visibilityBuffer = visBuffer;

  // first pass rts
  perframeData.m_visPassDepth = visDepth;
  perframeData.m_visColorRT = rhi->createRenderTarget(
      visBuffer.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Clear, 0, 0);
  perframeData.m_visDepthRT = rhi->createRenderTargetDepthStencil(
      visDepth, {{}, 1.0f}, RhiRenderTargetLoadOp::Clear);

  perframeData.m_visRTs = rhi->createRenderTargets();
  perframeData.m_visRTs->setColorAttachments({perframeData.m_visColorRT.get()});
  perframeData.m_visRTs->setDepthStencilAttachment(
      perframeData.m_visDepthRT.get());
  perframeData.m_visRTs->setRenderArea(renderTargets->getRenderArea());

  // second pass rts
  perframeData.m_visColorRT2 = rhi->createRenderTarget(
      visBuffer.get(), {{0, 0, 0, 0}}, RhiRenderTargetLoadOp::Load, 0, 0);
  perframeData.m_visDepthRT2 = rhi->createRenderTargetDepthStencil(
      visDepth, {{}, 1.0f}, RhiRenderTargetLoadOp::Load);
  perframeData.m_visRTs2 = rhi->createRenderTargets();
  perframeData.m_visRTs2->setColorAttachments(
      {perframeData.m_visColorRT2.get()});
  perframeData.m_visRTs2->setDepthStencilAttachment(
      perframeData.m_visDepthRT2.get());
  perframeData.m_visRTs2->setRenderArea(renderTargets->getRenderArea());

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
  uint32_t totalMeshlets = 0;
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

      for (int k = 0; k < perframeData.m_shaderEffectData[x].m_materials.size();
           k++) {
        auto mesh = perframeData.m_shaderEffectData[x].m_meshes[k]->loadMesh();
        totalMeshlets += mesh->m_meshlets.size();
      }
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
  if (perframeData.m_allFilteredMeshletsCount == nullptr) {
    perframeData.m_allFilteredMeshletsCount =
        rhi->createIndirectMeshDrawBufferDevice(
            2, RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT |
                   RhiBufferUsage::RHI_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  }
  if (perframeData.m_allFilteredMeshletsMaxCount < totalMeshlets) {
    perframeData.m_allFilteredMeshletsMaxCount = totalMeshlets;
    perframeData.m_allFilteredMeshlets = rhi->createStorageBufferDevice(
        totalMeshlets * sizeof(uint32_t) * 2,
        RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

    perframeData.m_allFilteredMeshletsDesc = rhi->createBindlessDescriptorRef();
    perframeData.m_allFilteredMeshletsDesc->addStorageBuffer(
        perframeData.m_allFilteredMeshlets, 0);
    perframeData.m_allFilteredMeshletsDesc->addStorageBuffer(
        perframeData.m_allFilteredMeshletsCount, 1);
  }
}

} // namespace Ifrit::Core