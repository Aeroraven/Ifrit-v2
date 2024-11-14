#include "ifrit/core/renderer/RendererBase.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {
IFRIT_APIDECL void RendererBase::buildPipelines(PerFrameData &perframeData,
                                                GraphicsShaderPassType passType,
                                                RenderTargets *renderTargets) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  for (auto &shaderEffectId : perframeData.m_enabledEffects) {
    auto &shaderEffect = perframeData.m_shaderEffectData[shaderEffectId];
    auto ref = shaderEffect.m_materials[0]->m_effectTemplates[passType];
    auto rtFormats = renderTargets->getFormat();
    PipelineAttachmentConfigs paConfig = {rtFormats.m_depthFormat,
                                          rtFormats.m_colorFormats};

    if (ref.m_drawPasses.count(paConfig) > 0) {
      continue;
    }
    RhiShader *vertexShader = nullptr, *fragmentShader = nullptr,
              *taskShader = nullptr, *meshShader = nullptr;
    auto rhi = m_app->getRhiLayer();
    auto pass = rhi->createGraphicsPass();
    uint32_t maxSets = 0;
    for (auto &shader : ref.m_shaders) {
      if (shader->getStage() == RhiShaderStage::Vertex) {
        vertexShader = shader;
        pass->setVertexShader(shader);
      } else if (shader->getStage() == RhiShaderStage::Fragment) {
        fragmentShader = shader;
        pass->setPixelShader(shader);
        maxSets = std::max(maxSets, shader->getNumDescriptorSets());
      } else if (shader->getStage() == RhiShaderStage::Task) {
        taskShader = shader;
      } else if (shader->getStage() == RhiShaderStage::Mesh) {
        meshShader = shader;
        pass->setMeshShader(shader);
        maxSets = std::max(maxSets, shader->getNumDescriptorSets());
      }
    }
    if (maxSets == 0) {
      throw std::runtime_error("No descriptor sets found in shader");
    }
    pass->setNumBindlessDescriptorSets(maxSets - 1);
    pass->setRenderTargetFormat(rtFormats);
    for (auto &material : shaderEffect.m_materials) {
      material->m_effectTemplates[passType].m_drawPasses[paConfig] = pass;
    }
  }
}

IFRIT_APIDECL void
RendererBase::prepareDeviceResources(PerFrameData &perframeData) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  auto rhi = m_app->getRhiLayer();

  std::vector<std::shared_ptr<RhiStagedSingleBuffer>> stagedBuffers;
  std::vector<void *> pendingVertexBuffers;
  std::vector<uint32_t> pendingVertexBufferSizes;

  if (perframeData.m_viewBindlessRef == nullptr) {
    perframeData.m_viewBuffer =
        rhi->createUniformBufferShared(sizeof(PerFramePerViewData), true, 0);
    perframeData.m_viewBindlessRef = rhi->createBindlessDescriptorRef();
    perframeData.m_viewBindlessRef->addUniformBuffer(perframeData.m_viewBuffer,
                                                     0);
  }

  // Update view buffer
  auto viewBuffer = perframeData.m_viewBuffer;
  auto viewBufferAct = viewBuffer->getActiveBuffer();
  viewBufferAct->map();
  viewBufferAct->writeBuffer(&perframeData.m_viewData,
                             sizeof(PerFramePerViewData), 0);
  viewBufferAct->flush();
  viewBufferAct->unmap();

  // Per effect data
  for (auto &shaderEffectId : perframeData.m_enabledEffects) {
    auto &shaderEffect = perframeData.m_shaderEffectData[shaderEffectId];
    // find whether batched object data should be recreated
    auto lastObjectCount = shaderEffect.m_lastObjectCount;
    auto objectCount = shaderEffect.m_materials.size();
    if (lastObjectCount != objectCount || lastObjectCount == ~0u) {
      // TODO/EMERGENCY: release old buffer
      shaderEffect.m_lastObjectCount = objectCount;
      shaderEffect.m_objectData.resize(objectCount);
      shaderEffect.m_batchedObjectData = rhi->createStorageBufferShared(
          sizeof(PerObjectData) * objectCount, true, 0);
      // TODO: update instead of recreate
      shaderEffect.m_batchedObjBufRef = rhi->createBindlessDescriptorRef();
      shaderEffect.m_batchedObjBufRef->addStorageBuffer(
          shaderEffect.m_batchedObjectData, 0);
    }

    for (int i = 0; i < shaderEffect.m_materials.size(); i++) {
      // Setup transform buffers
      auto transform = shaderEffect.m_transforms[i];
      RhiMultiBuffer *transformBuffer = nullptr;
      std::shared_ptr<RhiBindlessIdRef> bindlessRef = nullptr;
      transform->getGPUResource(transformBuffer, bindlessRef);
      if (transformBuffer == nullptr) {
        transformBuffer =
            rhi->createUniformBufferShared(sizeof(float4x4), true, 0);
        bindlessRef = rhi->registerUniformBuffer(transformBuffer);
        transform->setGPUResource(transformBuffer, bindlessRef);
      }
      // update uniform buffer, TODO: dirty flag
      float4x4 model = transform->getModelToWorldMatrix();
      auto buf = transformBuffer->getActiveBuffer();
      buf->map();
      buf->writeBuffer(&model, sizeof(float4x4), 0);
      buf->flush();
      buf->unmap();
      shaderEffect.m_objectData[i].transformRef = bindlessRef->getActiveId();

      // Setup mesh buffers
      auto mesh = shaderEffect.m_meshes[i];
      auto meshDataRef = mesh->loadMesh();
      Mesh::GPUResource meshResource;
      bool requireUpdate = false;

      mesh->getGPUResource(meshResource);
      if (meshResource.objectBufferId == nullptr || mesh->m_resourceDirty) {
        requireUpdate = true;
        mesh->m_resourceDirty = false;
        meshDataRef->m_cpCounter.totalBvhNodes = meshDataRef->m_bvhNodes.size();
        meshDataRef->m_cpCounter.totalLods = meshDataRef->m_maxLod;
        meshDataRef->m_cpCounter.totalNumClusters =
            meshDataRef->m_clusterGroups.size();

        meshResource.vertexBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_verticesAligned.size() * sizeof(ifloat4),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.bvhNodeBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_bvhNodes.size() *
                sizeof(MeshProcLib::ClusterLod::FlattenedBVHNode),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.clusterGroupBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_clusterGroups.size() *
                sizeof(MeshProcLib::ClusterLod::ClusterGroup),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshlets.size() * sizeof(iint4),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletVertexBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshletVertices.size() * sizeof(uint32_t),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletIndexBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshletTriangles.size() * sizeof(uint32_t),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletCullBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshCullData.size() *
                sizeof(MeshProcLib::ClusterLod::MeshletCullData),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletInClusterBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshletInClusterGroup.size() * sizeof(uint32_t),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.cpCounterBuffer = rhi->createStorageBufferDevice(
            sizeof(MeshData::GPUCPCounter), RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

        // Indices in bindless descriptors
        meshResource.vertexBufferId =
            rhi->registerStorageBuffer(meshResource.vertexBuffer);
        meshResource.bvhNodeBufferId =
            rhi->registerStorageBuffer(meshResource.bvhNodeBuffer);
        meshResource.clusterGroupBufferId =
            rhi->registerStorageBuffer(meshResource.clusterGroupBuffer);
        meshResource.meshletBufferId =
            rhi->registerStorageBuffer(meshResource.meshletBuffer);
        meshResource.meshletVertexBufferId =
            rhi->registerStorageBuffer(meshResource.meshletVertexBuffer);
        meshResource.meshletIndexBufferId =
            rhi->registerStorageBuffer(meshResource.meshletIndexBuffer);
        meshResource.meshletCullBufferId =
            rhi->registerStorageBuffer(meshResource.meshletCullBuffer);
        meshResource.meshletInClusterBufferId =
            rhi->registerStorageBuffer(meshResource.meshletInClusterBuffer);
        meshResource.cpCounterBufferId =
            rhi->registerStorageBuffer(meshResource.cpCounterBuffer);

        // Here, we assume that no double bufferring is allowed
        // meaning no CPU-GPU data transfer is allowed for mesh data after
        // initialization
        Mesh::GPUObjectBuffer &objectBuffer = meshResource.objectData;
        objectBuffer.vertexBufferId =
            meshResource.vertexBufferId->getActiveId();
        objectBuffer.bvhNodeBufferId =
            meshResource.bvhNodeBufferId->getActiveId();
        objectBuffer.clusterGroupBufferId =
            meshResource.clusterGroupBufferId->getActiveId();
        objectBuffer.meshletBufferId =
            meshResource.meshletBufferId->getActiveId();
        objectBuffer.meshletVertexBufferId =
            meshResource.meshletVertexBufferId->getActiveId();
        objectBuffer.meshletIndexBufferId =
            meshResource.meshletIndexBufferId->getActiveId();
        objectBuffer.meshletCullBufferId =
            meshResource.meshletCullBufferId->getActiveId();
        objectBuffer.meshletInClusterBufferId =
            meshResource.meshletInClusterBufferId->getActiveId();
        objectBuffer.cpCounterBufferId =
            meshResource.cpCounterBufferId->getActiveId();
        objectBuffer.boundingSphere =
            mesh->getBoundingSphere(meshDataRef->m_vertices);

        // description for the whole mesh
        meshResource.objectBuffer = rhi->createStorageBufferDevice(
            sizeof(Mesh::GPUObjectBuffer), RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.objectBufferId =
            rhi->registerStorageBuffer(meshResource.objectBuffer);

        mesh->setGPUResource(meshResource);
      }

      // Setup instance buffers
      auto meshInst = shaderEffect.m_instances[i];
      MeshInstance::GPUResource instanceResource;
      meshInst->getGPUResource(instanceResource);
      auto &meshInstObjData = meshInst->m_resource.objectData;
      if (instanceResource.objectBuffer == nullptr) {
        requireUpdate = true;
        instanceResource.cpQueueBuffer = rhi->createStorageBufferDevice(
            sizeof(uint32_t) * meshDataRef->m_bvhNodes.size(),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        instanceResource.cpCounterBuffer =
            rhi->createStorageBufferDevice(sizeof(MeshInstance::GPUCPCounter),
                                           RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        instanceResource.filteredMeshlets = rhi->createStorageBufferDevice(
            sizeof(uint32_t) * meshDataRef->m_meshlets.size(), 0);

        instanceResource.cpQueueBufferId =
            rhi->registerStorageBuffer(instanceResource.cpQueueBuffer);
        instanceResource.cpCounterBufferId =
            rhi->registerStorageBuffer(instanceResource.cpCounterBuffer);
        instanceResource.filteredMeshletsId =
            rhi->registerStorageBuffer(instanceResource.filteredMeshlets);

        instanceResource.objectData.cpCounterBufferId =
            instanceResource.cpCounterBufferId->getActiveId();
        instanceResource.objectData.cpQueueBufferId =
            instanceResource.cpQueueBufferId->getActiveId();
        instanceResource.objectData.filteredMeshletsId =
            instanceResource.filteredMeshletsId->getActiveId();

        instanceResource.objectBuffer = rhi->createStorageBufferDevice(
            sizeof(MeshInstance::GPUObjectBuffer),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        instanceResource.objectBufferId =
            rhi->registerStorageBuffer(instanceResource.objectBuffer);

        meshInst->setGPUResource(instanceResource);
      }

      shaderEffect.m_objectData[i].objectDataRef =
          meshResource.objectBufferId->getActiveId();
      shaderEffect.m_objectData[i].instanceDataRef =
          meshInst->m_resource.objectBufferId->getActiveId();

      // update vertex buffer, TODO: dirty flag
      if (requireUpdate) {
        auto funcEnqueueStagedBuffer =
            [&](std::shared_ptr<RhiStagedSingleBuffer> stagedBuffer, void *data,
                uint32_t size) {
              stagedBuffers.push_back(stagedBuffer);
              pendingVertexBuffers.push_back(data);
              pendingVertexBufferSizes.push_back(size);
            };
#define enqueueStagedBuffer(name, vecBuffer)                                   \
  auto staged##name = rhi->createStagedSingleBuffer(meshResource.name);        \
  funcEnqueueStagedBuffer(                                                     \
      staged##name, meshDataRef->vecBuffer.data(),                             \
      meshDataRef->vecBuffer.size() *                                          \
          sizeof(decltype(meshDataRef->vecBuffer)::value_type))

        enqueueStagedBuffer(vertexBuffer, m_verticesAligned);
        enqueueStagedBuffer(bvhNodeBuffer, m_bvhNodes);
        enqueueStagedBuffer(clusterGroupBuffer, m_clusterGroups);
        enqueueStagedBuffer(meshletBuffer, m_meshlets);
        enqueueStagedBuffer(meshletVertexBuffer, m_meshletVertices);
        enqueueStagedBuffer(meshletIndexBuffer, m_meshletTriangles);
        enqueueStagedBuffer(meshletCullBuffer, m_meshCullData);
        enqueueStagedBuffer(meshletInClusterBuffer, m_meshletInClusterGroup);

        auto stagedCPCounterBuffer =
            rhi->createStagedSingleBuffer(meshResource.cpCounterBuffer);
        stagedBuffers.push_back(stagedCPCounterBuffer);
        pendingVertexBuffers.push_back(&meshDataRef->m_cpCounter);
        pendingVertexBufferSizes.push_back(sizeof(MeshData::GPUCPCounter));

        auto stagedObjectBuffer =
            rhi->createStagedSingleBuffer(meshResource.objectBuffer);
        stagedBuffers.push_back(stagedObjectBuffer);
        pendingVertexBuffers.push_back(&mesh->m_resource.objectData);
        pendingVertexBufferSizes.push_back(sizeof(Mesh::GPUObjectBuffer));

        auto stagedInstanceObjectBuffer =
            rhi->createStagedSingleBuffer(instanceResource.objectBuffer);
        stagedBuffers.push_back(stagedInstanceObjectBuffer);
        pendingVertexBuffers.push_back(&meshInst->m_resource.objectData);
        pendingVertexBufferSizes.push_back(
            sizeof(MeshInstance::GPUObjectBuffer));

#undef enqueueStagedBuffer
      }
    }

    // update batched object data
    auto batchedObjectData = shaderEffect.m_batchedObjectData;
    auto batchedObjectDataAct = batchedObjectData->getActiveBuffer();
    batchedObjectDataAct->map();
    batchedObjectDataAct->writeBuffer(shaderEffect.m_objectData.data(),
                                      sizeof(PerObjectData) * objectCount, 0);
    batchedObjectDataAct->flush();
    batchedObjectDataAct->unmap();
  }
  // Issue a command buffer to copy data to GPU
  if (stagedBuffers.size() > 0) {
    auto queue = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    queue->runSyncCommand([&](const RhiCommandBuffer *cmd) {
      for (int i = 0; i < stagedBuffers.size(); i++) {
        stagedBuffers[i]->cmdCopyToDevice(cmd, pendingVertexBuffers[i],
                                          pendingVertexBufferSizes[i], 0);
      }
    });
  }
}

IFRIT_APIDECL void
RendererBase::endFrame(const std::vector<GPUCommandSubmission *> &cmdToWait) {
  auto rhi = m_app->getRhiLayer();
  using namespace Ifrit::GraphicsBackend;
  auto drawq = rhi->getQueue(Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto swapchainImg = rhi->getSwapchainImage();
  auto sRenderComplete = rhi->getSwapchainRenderDoneEventHandler();
  auto cmd = drawq->runAsyncCommand(
      [&](const Rhi::RhiCommandBuffer *cmd) {
        cmd->imageBarrier(swapchainImg, Rhi::RhiResourceState::Undefined,
                          Rhi::RhiResourceState::Present, {0, 0, 1, 1});
      },
      cmdToWait, {sRenderComplete.get()});
  rhi->endFrame();
}
IFRIT_APIDECL std::unique_ptr<RendererBase::GPUCommandSubmission>
RendererBase::beginFrame() {
  auto rhi = m_app->getRhiLayer();
  rhi->beginFrame();
  return rhi->getSwapchainFrameReadyEventHandler();
}

} // namespace Ifrit::Core