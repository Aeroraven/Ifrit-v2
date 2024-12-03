#include "ifrit/core/renderer/RendererBase.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {
IFRIT_APIDECL void RendererBase::collectPerframeData(
    PerFrameData &perframeData, Scene *scene, Camera *camera,
    GraphicsShaderPassType passType, const SceneCollectConfig &config) {
  using Ifrit::Common::Utility::size_cast;
  // Filling per frame data
  if (camera == nullptr) {
    camera = scene->getMainCamera();
  }
  if (camera == nullptr) {
    throw std::runtime_error("No camera found in scene");
  }
  perframeData.m_views.resize(1);
  auto &viewData = perframeData.m_views[0];
  viewData.m_viewType = PerFrameData::ViewType::Primary;
  viewData.m_viewDataOld = viewData.m_viewData;
  viewData.m_viewData.m_worldToView = camera->worldToCameraMatrix();
  viewData.m_viewData.m_perspective = camera->projectionMatrix();
  viewData.m_viewData.m_perspective[2][0] += config.projectionTranslateX;
  viewData.m_viewData.m_perspective[2][1] += config.projectionTranslateY;
  viewData.m_viewData.m_worldToClip = Math::transpose(
      Math::matmul(Math::transpose(viewData.m_viewData.m_perspective),
                   Math::transpose(viewData.m_viewData.m_worldToView)));
  viewData.m_viewData.m_cameraAspect = camera->getAspect();
  viewData.m_viewData.m_inversePerspective =
      Ifrit::Math::inverse4(viewData.m_viewData.m_perspective);
  viewData.m_viewData.m_clipToWorld =
      Math::inverse4(viewData.m_viewData.m_worldToClip);
  auto cameraTransform = camera->getParent()->getComponent<Transform>();
  if (cameraTransform == nullptr) {
    throw std::runtime_error("Camera has no transform");
  }
  auto pos = cameraTransform->getPosition();
  viewData.m_viewData.m_cameraPosition = ifloat4{pos.x, pos.y, pos.z, 1.0f};
  viewData.m_viewData.m_cameraFront = camera->getFront();
  viewData.m_viewData.m_cameraNear = camera->getNear();
  viewData.m_viewData.m_cameraFar = camera->getFar();
  viewData.m_viewData.m_cameraFovX = camera->getFov();
  viewData.m_viewData.m_cameraFovY = camera->getFov();

  // For each mesh renderer, get the material, mesh, and transform
  perframeData.m_enabledEffects.clear();
  for (auto &effect : perframeData.m_shaderEffectData) {
    effect.m_materials.clear();
    effect.m_meshes.clear();
    effect.m_transforms.clear();
    effect.m_instances.clear();
  }

  std::vector<std::shared_ptr<Material>> materials;
  std::vector<std::shared_ptr<Mesh>> meshes;
  std::vector<std::shared_ptr<Transform>> transforms;
  std::vector<std::shared_ptr<MeshInstance>> instances;

  std::vector<SceneNode *> nodes;
  nodes.push_back(scene->getRootNode().get());
  while (!nodes.empty()) {
    auto node = nodes.back();
    nodes.pop_back();
    for (auto &child : node->getChildren()) {
      nodes.push_back(child.get());
    }
    for (auto &obj : node->getGameObjects()) {
      auto meshRenderer = obj->getComponent<MeshRenderer>();
      auto meshFilter = obj->getComponent<MeshFilter>();
      auto transform = obj->getComponent<Transform>();
      if (!meshRenderer || !meshFilter || !transform) {
        continue;
      }
      if (meshRenderer && meshFilter && transform) {
        materials.push_back(meshRenderer->getMaterial());
        meshes.push_back(meshFilter->getMesh());
        transforms.push_back(transform);
        instances.push_back(meshFilter->getMeshInstance());
      } else {
        throw std::runtime_error(
            "MeshRenderer, MeshFilter, or Transform not found");
      }
    }
  }

  // Groups meshes with the same shader effect
  for (size_t i = 0; i < materials.size(); i++) {
    auto &material = materials[i];
    auto &mesh = meshes[i];
    auto &transform = transforms[i];
    auto &instance = instances[i];

    ShaderEffect effect;
    effect.m_shaders = material->m_effectTemplates[passType].m_shaders;

    // TODO: Heavy copy operation, should be avoided
    effect.m_drawPasses = material->m_effectTemplates[passType].m_drawPasses;
    if (perframeData.m_shaderEffectMap.count(effect) == 0) {
      perframeData.m_shaderEffectMap[effect] =
          size_cast<uint32_t>(perframeData.m_shaderEffectData.size());
      perframeData.m_shaderEffectData.push_back(PerShaderEffectData{});
    }
    auto id = perframeData.m_shaderEffectMap[effect];
    auto &shaderEffectData = perframeData.m_shaderEffectData[id];
    perframeData.m_enabledEffects.insert(id);

    shaderEffectData.m_materials.push_back(material);
    shaderEffectData.m_meshes.push_back(mesh);
    shaderEffectData.m_transforms.push_back(transform);
    shaderEffectData.m_instances.push_back(instance);
  }
}
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
RendererBase::recreateGBuffers(PerFrameData &perframeData,
                               RenderTargets *renderTargets) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  auto rhi = m_app->getRhiLayer();
  auto rtArea = renderTargets->getRenderArea();
  auto needRecreate = (perframeData.m_gbuffer.m_rtCreated == 0);
  if (!needRecreate) {
    needRecreate = (perframeData.m_gbuffer.m_rtWidth != rtArea.width ||
                    perframeData.m_gbuffer.m_rtHeight != rtArea.height);
  }
  if (needRecreate) {
    perframeData.m_gbuffer.m_rtCreated = 1;
    perframeData.m_gbuffer.m_rtWidth = rtArea.width;
    perframeData.m_gbuffer.m_rtHeight = rtArea.height;

    auto targetUsage = RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT |
                       RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT |
                       RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT;

    auto targetFomrat = RhiImageFormat::RHI_FORMAT_R8G8B8A8_UNORM;

    perframeData.m_gbuffer.m_albedo_materialFlags =
        rhi->createRenderTargetTexture(rtArea.width + rtArea.x,
                                       rtArea.height + rtArea.y, targetFomrat,
                                       targetUsage);
    perframeData.m_gbuffer.m_emissive = rhi->createRenderTargetTexture(
        rtArea.width + rtArea.x, rtArea.height + rtArea.y, targetFomrat,
        targetUsage);
    perframeData.m_gbuffer.m_normal_smoothness = rhi->createRenderTargetTexture(
        rtArea.width + rtArea.x, rtArea.height + rtArea.y, targetFomrat,
        targetUsage);
    perframeData.m_gbuffer.m_specular_occlusion =
        rhi->createRenderTargetTexture(rtArea.width + rtArea.x,
                                       rtArea.height + rtArea.y, targetFomrat,
                                       targetUsage);
    perframeData.m_gbuffer.m_shadowMask = rhi->createRenderTargetTexture(
        rtArea.width + rtArea.x, rtArea.height + rtArea.y, targetFomrat,
        targetUsage);

    // Then bindless ids
    perframeData.m_gbuffer.m_albedo_materialFlagsId = rhi->registerUAVImage(
        perframeData.m_gbuffer.m_albedo_materialFlags.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_emissiveId = rhi->registerUAVImage(
        perframeData.m_gbuffer.m_emissive.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_normal_smoothnessId = rhi->registerUAVImage(
        perframeData.m_gbuffer.m_normal_smoothness.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_specular_occlusionId = rhi->registerUAVImage(
        perframeData.m_gbuffer.m_specular_occlusion.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_shadowMaskId = rhi->registerUAVImage(
        perframeData.m_gbuffer.m_shadowMask.get(), {0, 0, 1, 1});

    // Then gbuffer refs
    PerFrameData::GBufferDesc gbufferDesc;
    gbufferDesc.m_albedo_materialFlags =
        perframeData.m_gbuffer.m_albedo_materialFlagsId->getActiveId();
    gbufferDesc.m_emissive = perframeData.m_gbuffer.m_emissiveId->getActiveId();
    gbufferDesc.m_normal_smoothness =
        perframeData.m_gbuffer.m_normal_smoothnessId->getActiveId();
    gbufferDesc.m_specular_occlusion =
        perframeData.m_gbuffer.m_specular_occlusionId->getActiveId();
    gbufferDesc.m_shadowMask =
        perframeData.m_gbuffer.m_shadowMaskId->getActiveId();

    // Then gbuffer desc
    perframeData.m_gbuffer.m_gbufferRefs = rhi->createStorageBufferDevice(
        sizeof(PerFrameData::GBufferDesc),
        RhiBufferUsage::RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto stagedBuf =
        rhi->createStagedSingleBuffer(perframeData.m_gbuffer.m_gbufferRefs);
    auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
      stagedBuf->cmdCopyToDevice(cmd, &gbufferDesc,
                                 sizeof(PerFrameData::GBufferDesc), 0);
    });
    perframeData.m_gbuffer.m_gbufferDesc = rhi->createBindlessDescriptorRef();
    perframeData.m_gbuffer.m_gbufferDesc->addStorageBuffer(
        perframeData.m_gbuffer.m_gbufferRefs, 0);

    // Then gbuffer desc for pixel shader
    perframeData.m_gbufferSampler = rhi->createTrivialSampler();
    perframeData.m_gbufferDescFrag = rhi->createBindlessDescriptorRef();
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(
        perframeData.m_gbuffer.m_albedo_materialFlags.get(),
        perframeData.m_gbufferSampler.get(), 0);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(
        perframeData.m_gbuffer.m_specular_occlusion.get(),
        perframeData.m_gbufferSampler.get(), 1);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(
        perframeData.m_gbuffer.m_normal_smoothness.get(),
        perframeData.m_gbufferSampler.get(), 2);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(
        perframeData.m_gbuffer.m_emissive.get(),
        perframeData.m_gbufferSampler.get(), 3);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(
        perframeData.m_gbuffer.m_shadowMask.get(),
        perframeData.m_gbufferSampler.get(), 4);

    // Create a uav barrier for the gbuffer
    perframeData.m_gbuffer.m_gbufferBarrier.clear();
    RhiResourceBarrier bAlbedo;
    bAlbedo.m_type = RhiBarrierType::UAVAccess;
    bAlbedo.m_uav.m_texture =
        perframeData.m_gbuffer.m_albedo_materialFlags.get();
    bAlbedo.m_uav.m_type = RhiResourceType::Texture;

    RhiResourceBarrier bEmissive;
    bEmissive.m_type = RhiBarrierType::UAVAccess;
    bEmissive.m_uav.m_texture = perframeData.m_gbuffer.m_emissive.get();
    bEmissive.m_uav.m_type = RhiResourceType::Texture;

    RhiResourceBarrier bNormal;
    bNormal.m_type = RhiBarrierType::UAVAccess;
    bNormal.m_uav.m_texture = perframeData.m_gbuffer.m_normal_smoothness.get();
    bNormal.m_uav.m_type = RhiResourceType::Texture;

    RhiResourceBarrier bSpecular;
    bSpecular.m_type = RhiBarrierType::UAVAccess;
    bSpecular.m_uav.m_texture =
        perframeData.m_gbuffer.m_specular_occlusion.get();
    bSpecular.m_uav.m_type = RhiResourceType::Texture;

    RhiResourceBarrier bShadow;
    bShadow.m_type = RhiBarrierType::UAVAccess;
    bShadow.m_uav.m_texture = perframeData.m_gbuffer.m_shadowMask.get();
    bShadow.m_uav.m_type = RhiResourceType::Texture;

    perframeData.m_gbuffer.m_gbufferBarrier.push_back(bAlbedo);
    perframeData.m_gbuffer.m_gbufferBarrier.push_back(bEmissive);
    perframeData.m_gbuffer.m_gbufferBarrier.push_back(bNormal);
    perframeData.m_gbuffer.m_gbufferBarrier.push_back(bSpecular);
    perframeData.m_gbuffer.m_gbufferBarrier.push_back(bShadow);
  }
}

IFRIT_APIDECL void
RendererBase::prepareDeviceResources(PerFrameData &perframeData,
                                     RenderTargets *renderTargets) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  auto rhi = m_app->getRhiLayer();
  auto renderArea = renderTargets->getRenderArea();

  if (perframeData.m_views.size() > 1) {
    printf("Warning: Multiple views are not supported yet\n");
    throw std::runtime_error("Multiple views are not supported yet");
  }

  auto &primaryView = perframeData.m_views[0];
  primaryView.m_viewType = PerFrameData::ViewType::Primary;
  primaryView.m_viewData.m_renderHeight = renderArea.height;
  primaryView.m_viewData.m_renderWidth = renderArea.width;
  primaryView.m_viewData.m_hizLods =
      static_cast<uint32_t>(std::floor(
          std::log2(std::max(renderArea.width, renderArea.height)))) +
      1;

  std::vector<std::shared_ptr<RhiStagedSingleBuffer>> stagedBuffers;
  std::vector<void *> pendingVertexBuffers;
  std::vector<uint32_t> pendingVertexBufferSizes;

  for (uint32_t i = 0; i < perframeData.m_views.size(); i++) {
    bool initLastFrameMatrix = false;
    auto &curView = perframeData.m_views[i];
    if (curView.m_viewBindlessRef == nullptr) {
      curView.m_viewBuffer =
          rhi->createUniformBufferShared(sizeof(PerFramePerViewData), true, 0);
      curView.m_viewBindlessRef = rhi->createBindlessDescriptorRef();
      curView.m_viewBindlessRef->addUniformBuffer(curView.m_viewBuffer, 0);

      curView.m_viewBufferLast =
          rhi->createUniformBufferShared(sizeof(PerFramePerViewData), true, 0);
      curView.m_viewBindlessRef->addUniformBuffer(curView.m_viewBufferLast, 1);
      initLastFrameMatrix = true;

      curView.m_viewBufferId = rhi->registerUniformBuffer(curView.m_viewBuffer);
    }

    // Update view buffer

    auto viewBuffer = curView.m_viewBuffer;
    auto viewBufferAct = viewBuffer->getActiveBuffer();
    viewBufferAct->map();
    viewBufferAct->writeBuffer(&curView.m_viewData, sizeof(PerFramePerViewData),
                               0);
    viewBufferAct->flush();
    viewBufferAct->unmap();

    // Init last's frame matrix for the first frame
    if (initLastFrameMatrix) {
      curView.m_viewDataOld = curView.m_viewData;
    }
    auto viewBufferLast = curView.m_viewBufferLast;
    auto viewBufferLastAct = viewBufferLast->getActiveBuffer();
    viewBufferLastAct->map();
    viewBufferLastAct->writeBuffer(&curView.m_viewDataOld,
                                   sizeof(PerFramePerViewData), 0);
    viewBufferLastAct->flush();
    viewBufferLastAct->unmap();
  }

  // Per effect data
  uint32_t curShaderMaterialId = 0;
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
      RhiMultiBuffer *transformBufferLast = nullptr;
      std::shared_ptr<RhiBindlessIdRef> bindlessRef = nullptr;
      std::shared_ptr<RhiBindlessIdRef> bindlessRefLast = nullptr;
      transform->getGPUResource(transformBuffer, transformBufferLast,
                                bindlessRef, bindlessRefLast);
      bool initLastFrameMatrix = false;
      if (transformBuffer == nullptr) {
        transformBuffer =
            rhi->createUniformBufferShared(sizeof(float4x4), true, 0);
        bindlessRef = rhi->registerUniformBuffer(transformBuffer);
        transform->setGPUResource(transformBuffer, transformBufferLast,
                                  bindlessRef, bindlessRefLast);
        initLastFrameMatrix = true;
        transformBufferLast =
            rhi->createUniformBufferShared(sizeof(float4x4), true, 0);
        bindlessRefLast = rhi->registerUniformBuffer(transformBufferLast);
        transform->setGPUResource(transformBuffer, transformBufferLast,
                                  bindlessRef, bindlessRefLast);
      }
      // update uniform buffer, TODO: dirty flag
      auto transformDirty = transform->getDirtyFlag();
      if (transformDirty.changed) {
        float4x4 model = transform->getModelToWorldMatrix();
        auto buf = transformBuffer->getActiveBuffer();
        buf->map();
        buf->writeBuffer(&model, sizeof(float4x4), 0);
        buf->flush();
        buf->unmap();
        shaderEffect.m_objectData[i].transformRef = bindlessRef->getActiveId();
        shaderEffect.m_objectData[i].transformRefLast =
            bindlessRefLast->getActiveId();
      }

      if (initLastFrameMatrix) {
        transform->onFrameCollecting();
      }
      if (initLastFrameMatrix || transformDirty.lastChanged) {
        float4x4 modelLast = transform->getModelToWorldMatrixLast();
        auto bufLast = transformBufferLast->getActiveBuffer();
        bufLast->map();
        bufLast->writeBuffer(&modelLast, sizeof(float4x4), 0);
        bufLast->flush();
        bufLast->unmap();
        transform->onFrameCollecting();
      }

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
        meshResource.normalBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_normalsAligned.size() * sizeof(ifloat4),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.uvBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_uvs.size() * sizeof(ifloat2),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.bvhNodeBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_bvhNodes.size() *
                sizeof(MeshProcLib::MeshProcess::FlattenedBVHNode),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.clusterGroupBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_clusterGroups.size() *
                sizeof(MeshProcLib::MeshProcess::ClusterGroup),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshlets.size() * sizeof(MeshData::MeshletData),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletVertexBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshletVertices.size() * sizeof(uint32_t),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletIndexBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshletTriangles.size() * sizeof(uint32_t),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletCullBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshCullData.size() *
                sizeof(MeshProcLib::MeshProcess::MeshletCullData),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.meshletInClusterBuffer = rhi->createStorageBufferDevice(
            meshDataRef->m_meshletInClusterGroup.size() * sizeof(uint32_t),
            RHI_BUFFER_USAGE_TRANSFER_DST_BIT);
        meshResource.cpCounterBuffer = rhi->createStorageBufferDevice(
            sizeof(MeshData::GPUCPCounter), RHI_BUFFER_USAGE_TRANSFER_DST_BIT);

        // Indices in bindless descriptors
        meshResource.vertexBufferId =
            rhi->registerStorageBuffer(meshResource.vertexBuffer);
        meshResource.normalBufferId =
            rhi->registerStorageBuffer(meshResource.normalBuffer);
        meshResource.uvBufferId =
            rhi->registerStorageBuffer(meshResource.uvBuffer);
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
        objectBuffer.normalBufferId =
            meshResource.normalBufferId->getActiveId();
        objectBuffer.uvBufferId = meshResource.uvBufferId->getActiveId();
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

        auto safeNumMeshlets = meshDataRef->m_numMeshletsEachLod[0] +
                               meshDataRef->m_numMeshletsEachLod[1];
        instanceResource.filteredMeshlets = rhi->createStorageBufferDevice(
            sizeof(uint32_t) * safeNumMeshlets, 0);

        instanceResource.cpQueueBufferId =
            rhi->registerStorageBuffer(instanceResource.cpQueueBuffer);
        instanceResource.filteredMeshletsId =
            rhi->registerStorageBuffer(instanceResource.filteredMeshlets);

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
      shaderEffect.m_objectData[i].materialId = curShaderMaterialId;

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
        enqueueStagedBuffer(normalBuffer, m_normalsAligned);
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

    curShaderMaterialId++;
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
RendererBase::updateLastFrameTransforms(PerFrameData &perframeData) {
  throw std::runtime_error("Deprecated");
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
        cmd->imageBarrier(swapchainImg, Rhi::RhiResourceState::RenderTarget,
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