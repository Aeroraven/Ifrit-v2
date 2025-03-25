
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

#include "ifrit/core/renderer/RendererBase.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/util/Parallel.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/base/Light.h"
#include "ifrit/rhi/common/RhiLayer.h"

#include "ifrit/core/renderer/util/NoiseUtils.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"

#include <mutex>

using Ifrit::Common::Utility::size_cast;
namespace Ifrit::Core {

// ImmutableRendererResources RendererBase::m_immRes = {};

IFRIT_APIDECL void RendererBase::prepareImmutableResources() {
  if (m_immRes.m_initialized) {
    return;
  }
  m_immRes.m_initialized = true;
  std::lock_guard<std::mutex> lock(m_immRes.m_mutex);
  auto rhi = m_app->getRhiLayer();
  if (m_immRes.m_linearSampler == nullptr) {
    m_immRes.m_linearSampler = rhi->createTrivialSampler();
    m_immRes.m_nearestSampler = rhi->createTrivialNearestSampler(false);
  }

  // Load blue noise texture
  if (m_immRes.m_blueNoise == nullptr) {
    m_immRes.m_blueNoise = RenderingUtil::loadBlueNoise(rhi);
    m_immRes.m_blueNoiseSRV =
        rhi->registerCombinedImageSampler(m_immRes.m_blueNoise.get(), m_immRes.m_linearSampler.get());
  }
}

IFRIT_APIDECL void RendererBase::collectPerframeData(PerFrameData &perframeData, Scene *scene, Camera *camera,
                                                     GraphicsShaderPassType passType, RenderTargets *renderTargets,
                                                     const SceneCollectConfig &config) {
  using Ifrit::Common::Utility::size_cast;

  Logging::assertion(m_config != nullptr, "Renderer config is not set");
  // Filling per frame data
  if (camera == nullptr) {
    camera = scene->getMainCamera();
  }
  if (camera == nullptr) {
    throw std::runtime_error("No camera found in scene");
  }
  if (perframeData.m_views.size() == 0) {
    perframeData.m_views.resize(1);
  }
  if (true) {
    auto rtArea = renderTargets->getRenderArea();
    u32 actualRtWidth = 0, actualRtHeight = 0;
    getSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

    // Check if camera changed
    bool camChanged = false;

    auto &viewData = perframeData.m_views[0];
    if (true) {
      auto &viewData = perframeData.m_views[0];
      auto worldToView = Math::transpose(camera->worldToCameraMatrix());
      auto worldToViewLast = viewData.m_viewData.m_worldToView;
      for (auto i = 0; i < 4; i++) {
        for (auto j = 0; j < 4; j++) {
          if (worldToView[i][j] != worldToViewLast[i][j]) {
            camChanged = true;
            break;
          }
        }
      }
    }
    viewData.m_camMoved = camChanged;
    viewData.m_viewType = PerFrameData::ViewType::Primary;
    viewData.m_viewDataOld = viewData.m_viewData;
    viewData.m_viewData.m_worldToView = Math::transpose(camera->worldToCameraMatrix());
    viewData.m_viewData.m_perspective = Math::transpose(camera->projectionMatrix());
    viewData.m_viewData.m_perspective[2][0] += config.projectionTranslateX;
    viewData.m_viewData.m_perspective[2][1] += config.projectionTranslateY;
    viewData.m_viewData.m_worldToClip = Math::transpose(Math::matmul(
        Math::transpose(viewData.m_viewData.m_perspective), Math::transpose(viewData.m_viewData.m_worldToView)));
    viewData.m_viewData.m_viewToWorld =
        Math::transpose(Math::inverse4(Math::transpose(viewData.m_viewData.m_worldToView)));
    viewData.m_viewData.m_cameraAspect = camera->getAspect();
    viewData.m_viewData.m_inversePerspective =
        Math::transpose(Ifrit::Math::inverse4(Math::transpose(viewData.m_viewData.m_perspective)));
    viewData.m_viewData.m_clipToWorld =
        Math::transpose(Math::inverse4(Math::transpose(viewData.m_viewData.m_worldToClip)));
    auto cameraTransform = camera->getParent()->getComponentUnsafe<Transform>();
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
    viewData.m_viewData.m_cameraOrthoSize = camera->getOrthoSpaceSize();
    viewData.m_viewData.m_viewCameraType = camera->getCameraType() == CameraType::Orthographic ? 1.0f : 0.0f;
  }

  // Find lights that represent the sun
  auto sunLights = scene->filterObjectsUnsafe([](SceneObject *obj) {
    auto light = obj->getComponentUnsafe<Light>();
    if (!light) {
      return false;
    }
    return light->getAffectPbrSky();
  });
  if (sunLights.size() > 1) {
    iError("Multiple sun lights found");
    throw std::runtime_error("Multiple sun lights found");
  }
  if (sunLights.size() == 1) {
    auto transform = sunLights[0]->getComponentUnsafe<Transform>();
    ifloat4 front = {0.0f, 0.0f, 1.0f, 0.0f};
    auto rotation = transform->getRotation();
    auto rotMatrix = Math::eulerAngleToMatrix(rotation);
    front = Math::matmul(rotMatrix, front);
    // front.z = -front.z;
    perframeData.m_sunDir = front;
  }

  // Insert light view data, if shadow maps are enabled
  auto lightWithShadow = scene->filterObjectsUnsafe([](SceneObject *obj) -> bool {
    auto light = obj->getComponentUnsafe<Light>();
    if (!light) {
      return false;
    }
    auto shadow = light->getShadowMap();
    return shadow;
  });

  auto shadowViewCounts = size_cast<u32>(lightWithShadow.size()) * m_config->m_shadowConfig.m_csmCount;

  if (perframeData.m_views.size() < 1 + shadowViewCounts) {
    perframeData.m_views.resize(1 + shadowViewCounts);
  }
  if (perframeData.m_shadowData2.m_shadowViews.size() < m_config->m_shadowConfig.k_maxShadowMaps) {
    perframeData.m_shadowData2.m_shadowViews.resize(m_config->m_shadowConfig.k_maxShadowMaps);
  }
  perframeData.m_shadowData2.m_enabledShadowMaps = size_cast<u32>(lightWithShadow.size());
  for (auto di = 0, dj = 0; auto &lightObj : lightWithShadow) {
    // Temporarily, we assume that all lights are directional lights
    auto light = lightObj->getComponentUnsafe<Light>();
    auto lightTransform = lightObj->getComponentUnsafe<Transform>();
    std::vector<float> csmSplits(m_config->m_shadowConfig.m_csmSplits.begin(),
                                 m_config->m_shadowConfig.m_csmSplits.end());
    std::vector<float> csmBorders(m_config->m_shadowConfig.m_csmBorders.begin(),
                                  m_config->m_shadowConfig.m_csmBorders.end());
    auto maxDist = m_config->m_shadowConfig.m_maxDistance;
    std::array<float, 4> splitStart, splitEnd;
    auto &viewData = perframeData.m_views[0];
    auto csmViews = RenderingUtil::CascadeShadowMapping::fillCSMViews(
        viewData, *light, light->getShadowMapResolution(), *lightTransform, m_config->m_shadowConfig.m_csmCount,
        maxDist, csmSplits, csmBorders, splitStart, splitEnd);

    perframeData.m_shadowData2.m_shadowViews[di].m_csmSplits = m_config->m_shadowConfig.m_csmCount;
    perframeData.m_shadowData2.m_shadowViews[di].m_csmStart = splitStart;
    perframeData.m_shadowData2.m_shadowViews[di].m_csmEnd = splitEnd;
    for (auto i = 0; auto &csmView : csmViews) {
      perframeData.m_views[1 + dj].m_viewData = csmView.m_viewData;
      perframeData.m_views[1 + dj].m_renderHeight = static_cast<u32>(csmView.m_viewData.m_renderHeightf);
      perframeData.m_views[1 + dj].m_renderWidth = static_cast<u32>(csmView.m_viewData.m_renderWidthf);
      perframeData.m_views[1 + dj].m_viewType = PerFrameData::ViewType::Shadow;
      perframeData.m_shadowData2.m_shadowViews[di].m_viewMapping[i] = dj + 1;
      dj++;
      i++;
    }
    di++;
  }
  // filling shadow data
  auto start = std::chrono::high_resolution_clock::now();
  // For each mesh renderer, get the material, mesh, and transform
  perframeData.m_enabledEffects.clear();
  for (auto &effect : perframeData.m_shaderEffectData) {
    effect.m_materials.clear();
    effect.m_meshes.clear();
    effect.m_transforms.clear();
    effect.m_instances.clear();
  }

  std::vector<Material *> materials;
  std::vector<Mesh *> meshes;
  std::vector<Transform *> transforms;
  std::vector<MeshInstance *> instances;

  std::vector<SceneNode *> nodes;
  nodes.push_back(scene->getRootNode().get());
  while (!nodes.empty()) {
    auto node = nodes.back();
    nodes.pop_back();
    for (auto &child : node->getChildren()) {
      nodes.push_back(child.get());
    }
    for (auto &obj : node->getGameObjects()) {
      auto meshRenderer = obj->getComponentUnsafe<MeshRenderer>();
      auto meshFilter = obj->getComponentUnsafe<MeshFilter>();
      if (!meshRenderer || !meshFilter) {
        continue;
      }
      auto transform = obj->getComponent<Transform>();
      if (meshRenderer && meshFilter && transform) {
        materials.push_back(meshRenderer->getMaterial().get());
        meshes.push_back(meshFilter->getMesh().get());
        transforms.push_back(transform.get());
        instances.push_back(meshFilter->getMeshInstance().get());
      } else {
        throw std::runtime_error("MeshRenderer, MeshFilter, or Transform not found");
      }
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
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
    effect.m_computePass = material->m_effectTemplates[passType].m_computePass;
    effect.m_type = material->m_effectTemplates[passType].m_type;

    if (perframeData.m_shaderEffectMap.count(effect) == 0) {
      perframeData.m_shaderEffectMap[effect] = size_cast<u32>(perframeData.m_shaderEffectData.size());
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

  auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  // iDebug("CPU time, Collecting per frame data, shader effects: {} ms",
  //       elapsed.count() / 1000.0f);
  return;
}
IFRIT_APIDECL void RendererBase::buildPipelines(PerFrameData &perframeData, GraphicsShaderPassType passType,
                                                RenderTargets *renderTargets) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  for (auto &shaderEffectId : perframeData.m_enabledEffects) {
    auto &shaderEffect = perframeData.m_shaderEffectData[shaderEffectId];
    auto ref = shaderEffect.m_materials[0]->m_effectTemplates[passType];

    if (ref.m_type == ShaderEffectType::Compute) {
      if (ref.m_computePass) {
        continue;
      }
      auto rhi = m_app->getRhiLayer();
      auto computePass = rhi->createComputePass();
      auto shader = ref.m_shaders[0];
      computePass->setComputeShader(shader);

      // TODO: obtain via reflection
      computePass->setPushConstSize(64); // Test value
      computePass->setNumBindlessDescriptorSets(3);
      for (auto &material : shaderEffect.m_materials) {
        material->m_effectTemplates[passType].m_computePass = computePass;
      }
      continue;
    }
    auto rtFormats = renderTargets->getFormat();
    PipelineAttachmentConfigs paConfig = {rtFormats.m_depthFormat, rtFormats.m_colorFormats};

    if (ref.m_drawPasses.count(paConfig) > 0) {
      continue;
    }
    RhiShader *vertexShader = nullptr, *fragmentShader = nullptr, *taskShader = nullptr, *meshShader = nullptr;
    auto rhi = m_app->getRhiLayer();
    auto pass = rhi->createGraphicsPass();
    u32 maxSets = 0;
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

IFRIT_APIDECL void RendererBase::recreateGBuffers(PerFrameData &perframeData, RenderTargets *renderTargets) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  auto rhi = m_app->getRhiLayer();
  auto rtArea = renderTargets->getRenderArea();
  auto needRecreate = (perframeData.m_gbuffer.m_rtCreated == 0);

  u32 actualRtWidth = 0, actualRtHeight = 0;
  getSupersampledRenderArea(renderTargets, &actualRtWidth, &actualRtHeight);

  if (!needRecreate) {
    needRecreate =
        (perframeData.m_gbuffer.m_rtWidth != actualRtWidth || perframeData.m_gbuffer.m_rtHeight != actualRtHeight);
  }
  if (needRecreate) {
    perframeData.m_gbuffer.m_rtCreated = 1;
    perframeData.m_gbuffer.m_rtWidth = actualRtWidth;
    perframeData.m_gbuffer.m_rtHeight = actualRtHeight;

    auto targetUsage = RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT |
                       RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT;

    // auto targetFomrat = RhiImageFormat::RHI_FORMAT_R8G8B8A8_UNORM;
    auto targetFomrat = RhiImageFormat::RHI_FORMAT_R32G32B32A32_SFLOAT;
    perframeData.m_gbuffer.m_albedo_materialFlags =
        rhi->createTexture2D("Render_GAlbedo", actualRtWidth, actualRtHeight, targetFomrat, targetUsage);
    perframeData.m_gbuffer.m_emissive =
        rhi->createTexture2D("Render_GEmissive", actualRtWidth, actualRtHeight, targetFomrat, targetUsage);
    perframeData.m_gbuffer.m_normal_smoothness =
        rhi->createTexture2D("Render_GNormSmooth", actualRtWidth, actualRtHeight, targetFomrat, targetUsage);
    perframeData.m_gbuffer.m_specular_occlusion =
        rhi->createTexture2D("Render_GSpecOccl", actualRtWidth, actualRtHeight, targetFomrat,
                             targetUsage | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    perframeData.m_gbuffer.m_specular_occlusion_intermediate =
        rhi->createTexture2D("Render_GSpecOccl_Intm", actualRtWidth, actualRtHeight, targetFomrat,
                             targetUsage | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    perframeData.m_gbuffer.m_shadowMask =
        rhi->createTexture2D("Render_GShadowMask", actualRtWidth, actualRtHeight, targetFomrat, targetUsage);

    // Then bindless ids
    perframeData.m_gbuffer.m_albedo_materialFlagsId =
        rhi->registerUAVImage(perframeData.m_gbuffer.m_albedo_materialFlags.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_emissiveId = rhi->registerUAVImage(perframeData.m_gbuffer.m_emissive.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_normal_smoothnessId =
        rhi->registerUAVImage(perframeData.m_gbuffer.m_normal_smoothness.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_specular_occlusionId =
        rhi->registerUAVImage(perframeData.m_gbuffer.m_specular_occlusion.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_specular_occlusion_intermediateId =
        rhi->registerUAVImage(perframeData.m_gbuffer.m_specular_occlusion_intermediate.get(), {0, 0, 1, 1});
    perframeData.m_gbuffer.m_shadowMaskId =
        rhi->registerUAVImage(perframeData.m_gbuffer.m_shadowMask.get(), {0, 0, 1, 1});

    // sampler
    perframeData.m_gbuffer.m_albedo_materialFlags_sampId = rhi->registerCombinedImageSampler(
        perframeData.m_gbuffer.m_albedo_materialFlags.get(), m_immRes.m_linearSampler.get());
    perframeData.m_gbuffer.m_normal_smoothness_sampId = rhi->registerCombinedImageSampler(
        perframeData.m_gbuffer.m_normal_smoothness.get(), m_immRes.m_linearSampler.get());
    perframeData.m_gbuffer.m_specular_occlusion_sampId = rhi->registerCombinedImageSampler(
        perframeData.m_gbuffer.m_specular_occlusion.get(), m_immRes.m_linearSampler.get());
    perframeData.m_gbuffer.m_specular_occlusion_intermediate_sampId = rhi->registerCombinedImageSampler(
        perframeData.m_gbuffer.m_specular_occlusion_intermediate.get(), m_immRes.m_linearSampler.get());

    // color rts
    RenderingUtil::warpRenderTargets(rhi, perframeData.m_gbuffer.m_specular_occlusion.get(),
                                     perframeData.m_gbuffer.m_specular_occlusion_colorRT,
                                     perframeData.m_gbuffer.m_specular_occlusion_RTs);

    // barriers
    perframeData.m_gbuffer.m_normal_smoothnessBarrier.m_type = RhiBarrierType::UAVAccess;
    perframeData.m_gbuffer.m_normal_smoothnessBarrier.m_uav.m_texture =
        perframeData.m_gbuffer.m_normal_smoothness.get();
    perframeData.m_gbuffer.m_normal_smoothnessBarrier.m_uav.m_type = RhiResourceType::Texture;

    perframeData.m_gbuffer.m_specular_occlusionBarrier.m_type = RhiBarrierType::UAVAccess;
    perframeData.m_gbuffer.m_specular_occlusionBarrier.m_uav.m_texture =
        perframeData.m_gbuffer.m_specular_occlusion.get();
    perframeData.m_gbuffer.m_specular_occlusionBarrier.m_uav.m_type = RhiResourceType::Texture;

    // Then gbuffer refs
    PerFrameData::GBufferDesc gbufferDesc;
    gbufferDesc.m_albedo_materialFlags = perframeData.m_gbuffer.m_albedo_materialFlagsId->getActiveId();
    gbufferDesc.m_emissive = perframeData.m_gbuffer.m_emissiveId->getActiveId();
    gbufferDesc.m_normal_smoothness = perframeData.m_gbuffer.m_normal_smoothnessId->getActiveId();
    gbufferDesc.m_specular_occlusion = perframeData.m_gbuffer.m_specular_occlusionId->getActiveId();
    gbufferDesc.m_shadowMask = perframeData.m_gbuffer.m_shadowMaskId->getActiveId();

    // Then gbuffer desc
    perframeData.m_gbuffer.m_gbufferRefs =
        rhi->createBufferDevice("Render_GbufferRef", sizeof(PerFrameData::GBufferDesc),
                                RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_SSBO);
    auto stagedBuf = rhi->createStagedSingleBuffer(perframeData.m_gbuffer.m_gbufferRefs.get());
    auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    tq->runSyncCommand([&](const RhiCommandList *cmd) {
      stagedBuf->cmdCopyToDevice(cmd, &gbufferDesc, sizeof(PerFrameData::GBufferDesc), 0);
    });
    perframeData.m_gbuffer.m_gbufferDesc = rhi->createBindlessDescriptorRef();
    perframeData.m_gbuffer.m_gbufferDesc->addStorageBuffer(perframeData.m_gbuffer.m_gbufferRefs.get(), 0);

    // Then gbuffer desc for pixel shader
    perframeData.m_gbufferDescFrag = rhi->createBindlessDescriptorRef();
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(perframeData.m_gbuffer.m_albedo_materialFlags.get(),
                                                            m_immRes.m_linearSampler.get(), 0);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(perframeData.m_gbuffer.m_specular_occlusion.get(),
                                                            m_immRes.m_linearSampler.get(), 1);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(perframeData.m_gbuffer.m_normal_smoothness.get(),
                                                            m_immRes.m_linearSampler.get(), 2);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(perframeData.m_gbuffer.m_emissive.get(),
                                                            m_immRes.m_linearSampler.get(), 3);
    perframeData.m_gbufferDescFrag->addCombinedImageSampler(perframeData.m_gbuffer.m_shadowMask.get(),
                                                            m_immRes.m_linearSampler.get(), 4);

    // Create a uav barrier for the gbuffer
    perframeData.m_gbuffer.m_gbufferBarrier.clear();
    RhiResourceBarrier bAlbedo;
    bAlbedo.m_type = RhiBarrierType::UAVAccess;
    bAlbedo.m_uav.m_texture = perframeData.m_gbuffer.m_albedo_materialFlags.get();
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
    bSpecular.m_uav.m_texture = perframeData.m_gbuffer.m_specular_occlusion.get();
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

IFRIT_APIDECL void RendererBase::prepareDeviceResources(PerFrameData &perframeData, RenderTargets *renderTargets) {
  using namespace Ifrit::GraphicsBackend::Rhi;
  auto rhi = m_app->getRhiLayer();
  auto renderArea = renderTargets->getRenderArea();

  u32 actualRenderWidth = 0;
  u32 actualRenderHeight = 0;
  getSupersampledRenderArea(renderTargets, &actualRenderWidth, &actualRenderHeight);

  auto &primaryView = perframeData.m_views[0];
  primaryView.m_viewType = PerFrameData::ViewType::Primary;
  primaryView.m_viewData.m_renderHeightf = static_cast<float>(actualRenderHeight);
  primaryView.m_viewData.m_renderWidthf = static_cast<float>(actualRenderWidth);
  primaryView.m_renderHeight = actualRenderHeight;
  primaryView.m_renderWidth = actualRenderWidth;
  primaryView.m_viewData.m_hizLods =
      std::floor(std::log2(static_cast<float>(std::max(actualRenderWidth, actualRenderHeight)))) + 1.0f;

  // Shadow views data has been filled in collectPerframeData

  std::vector<std::shared_ptr<RhiStagedSingleBuffer>> stagedBuffers;
  std::vector<void *> pendingVertexBuffers;
  std::vector<u32> pendingVertexBufferSizes;

  for (u32 i = 0; i < perframeData.m_views.size(); i++) {
    bool initLastFrameMatrix = false;
    auto &curView = perframeData.m_views[i];
    if (curView.m_viewBindlessRef == nullptr) {
      curView.m_viewBuffer =
          rhi->createBufferCoherent(sizeof(PerFramePerViewData), RhiBufferUsage::RhiBufferUsage_Uniform);
      curView.m_viewBindlessRef = rhi->createBindlessDescriptorRef();
      curView.m_viewBindlessRef->addUniformBuffer(curView.m_viewBuffer.get(), 0);
      curView.m_viewBufferLast =
          rhi->createBufferCoherent(sizeof(PerFramePerViewData), RhiBufferUsage::RhiBufferUsage_Uniform);

      curView.m_viewBindlessRef->addUniformBuffer(curView.m_viewBufferLast.get(), 1);
      initLastFrameMatrix = true;

      curView.m_viewBufferId = rhi->registerUniformBuffer(curView.m_viewBuffer.get());
    }

    // Update view buffer

    auto viewBuffer = curView.m_viewBuffer;
    auto viewBufferAct = viewBuffer->getActiveBuffer();
    viewBufferAct->map();
    viewBufferAct->writeBuffer(&curView.m_viewData, sizeof(PerFramePerViewData), 0);
    viewBufferAct->flush();
    viewBufferAct->unmap();

    // Init last's frame matrix for the first frame
    if (initLastFrameMatrix) {
      curView.m_viewDataOld = curView.m_viewData;
    }
    auto viewBufferLast = curView.m_viewBufferLast;
    auto viewBufferLastAct = viewBufferLast->getActiveBuffer();
    viewBufferLastAct->map();
    viewBufferLastAct->writeBuffer(&curView.m_viewDataOld, sizeof(PerFramePerViewData), 0);
    viewBufferLastAct->flush();
    viewBufferLastAct->unmap();
  }

  // Per effect data
  u32 curShaderMaterialId = 0;
  for (auto &shaderEffectId : perframeData.m_enabledEffects) {
    auto &shaderEffect = perframeData.m_shaderEffectData[shaderEffectId];
    // find whether batched object data should be recreated
    auto lastObjectCount = shaderEffect.m_lastObjectCount;
    auto objectCount = size_cast<u32>(shaderEffect.m_materials.size());
    if (lastObjectCount != objectCount || lastObjectCount == ~0u) {
      // TODO/EMERGENCY: release old buffer
      shaderEffect.m_lastObjectCount = objectCount;
      shaderEffect.m_objectData.resize(objectCount);
      shaderEffect.m_batchedObjectData =
          rhi->createBufferCoherent(sizeof(PerObjectData) * objectCount, RhiBufferUsage::RhiBufferUsage_SSBO);

      // TODO: update instead of recreate
      shaderEffect.m_batchedObjBufRef = rhi->createBindlessDescriptorRef();
      shaderEffect.m_batchedObjBufRef->addStorageBuffer(shaderEffect.m_batchedObjectData.get(), 0);
    }

    // preloading meshes
    Ifrit::Common::Utility::unordered_for<size_t>(0, shaderEffect.m_materials.size(),
                                                  [&](size_t x) { auto mesh = shaderEffect.m_meshes[x]; });

    // load meshes
    for (int i = 0; i < shaderEffect.m_materials.size(); i++) {
      // Setup transform buffers
      auto transform = shaderEffect.m_transforms[i];
      std::shared_ptr<RhiMultiBuffer> transformBuffer = nullptr;
      std::shared_ptr<RhiMultiBuffer> transformBufferLast = nullptr;
      std::shared_ptr<RhiBindlessIdRef> bindlessRef = nullptr;
      std::shared_ptr<RhiBindlessIdRef> bindlessRefLast = nullptr;
      transform->getGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
      bool initLastFrameMatrix = false;
      if (transformBuffer == nullptr) {
        transformBuffer =
            rhi->createBufferCoherent(sizeof(MeshInstanceTransform), RhiBufferUsage::RhiBufferUsage_Uniform);
        bindlessRef = rhi->registerUniformBuffer(transformBuffer.get());
        transform->setGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
        initLastFrameMatrix = true;
        transformBufferLast =
            rhi->createBufferCoherent(sizeof(MeshInstanceTransform), RhiBufferUsage::RhiBufferUsage_Uniform);
        bindlessRefLast = rhi->registerUniformBuffer(transformBufferLast.get());
        transform->setGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
      }

      // update uniform buffer, TODO: dirty flag
      auto transformDirty = transform->getDirtyFlag();
      if (transformDirty.changed || transformDirty.lastChanged) {
        MeshInstanceTransform model;
        // Transpose is required because glsl uses column major matrices
        model.model = Math::transpose(transform->getModelToWorldMatrix());
        model.invModel = Math::transpose(Math::inverse4(transform->getModelToWorldMatrix()));
        auto scale = transform->getScale();
        model.maxScale = std::max(scale.x, std::max(scale.y, scale.z));

        auto buf = transformBuffer->getActiveBuffer();
        buf->map();
        buf->writeBuffer(&model, sizeof(MeshInstanceTransform), 0);
        buf->flush();
        buf->unmap();
        shaderEffect.m_objectData[i].transformRef = bindlessRef->getActiveId();
        shaderEffect.m_objectData[i].transformRefLast = bindlessRefLast->getActiveId();
      }

      if (initLastFrameMatrix) {
        // transform->onFrameCollecting();
      }

      if (initLastFrameMatrix || transformDirty.lastChanged) {
        MeshInstanceTransform modelLast;
        modelLast.model = Math::transpose(transform->getModelToWorldMatrix());
        modelLast.invModel = Math::transpose(Math::inverse4(transform->getModelToWorldMatrix()));
        auto lastScale = transform->getScaleLast();
        modelLast.maxScale = std::max(lastScale.x, std::max(lastScale.y, lastScale.z));
        auto bufLast = transformBufferLast->getActiveBuffer();
        bufLast->map();
        bufLast->writeBuffer(&modelLast, sizeof(MeshInstanceTransform), 0);
        bufLast->flush();
        bufLast->unmap();
        transform->onFrameCollecting();
      }

      // Setup mesh buffers
      auto mesh = shaderEffect.m_meshes[i];
      auto meshDataRef = mesh->loadMesh();
      Mesh::GPUResource meshResource;
      bool requireUpdate = false;
      bool haveMaterialData = false;

      mesh->getGPUResource(meshResource);
      if (meshResource.objectBufferId == nullptr || mesh->m_resourceDirty) {
        requireUpdate = true;
        mesh->m_resourceDirty = false;
        meshDataRef->m_cpCounter.totalBvhNodes = size_cast<u32>(meshDataRef->m_bvhNodes.size());
        meshDataRef->m_cpCounter.totalLods = meshDataRef->m_maxLod;
        meshDataRef->m_cpCounter.totalNumClusters = size_cast<u32>(meshDataRef->m_clusterGroups.size());

        auto tmpUsage = RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;
        meshResource.vertexBuffer = rhi->createBufferDevice(
            "Mesh_Vertex", size_cast<u32>(meshDataRef->m_verticesAligned.size() * sizeof(ifloat4)), tmpUsage);
        meshResource.normalBuffer = rhi->createBufferDevice(
            "Mesh_Normal", size_cast<u32>(meshDataRef->m_normalsAligned.size() * sizeof(ifloat4)), tmpUsage);
        meshResource.uvBuffer =
            rhi->createBufferDevice("Mesh_UV", size_cast<u32>(meshDataRef->m_uvs.size() * sizeof(ifloat2)), tmpUsage);
        meshResource.bvhNodeBuffer = rhi->createBufferDevice(
            "Mesh_BVHNode",
            size_cast<u32>(meshDataRef->m_bvhNodes.size() * sizeof(MeshProcLib::MeshProcess::FlattenedBVHNode)),
            tmpUsage);
        meshResource.clusterGroupBuffer = rhi->createBufferDevice(
            "Mesh_ClusterGroup",
            size_cast<u32>(meshDataRef->m_clusterGroups.size() * sizeof(MeshProcLib::MeshProcess::ClusterGroup)),
            tmpUsage);
        meshResource.meshletBuffer = rhi->createBufferDevice(
            "Mesh_Cluster", size_cast<u32>(meshDataRef->m_meshlets.size() * sizeof(MeshData::MeshletData)), tmpUsage);
        meshResource.meshletVertexBuffer = rhi->createBufferDevice(
            "Mesh_ClusterVertex", size_cast<u32>(meshDataRef->m_meshletVertices.size() * sizeof(u32)), tmpUsage);
        meshResource.meshletIndexBuffer = rhi->createBufferDevice(
            "Mesh_ClusterIndex", size_cast<u32>(meshDataRef->m_meshletTriangles.size() * sizeof(u32)), tmpUsage);
        meshResource.meshletInClusterBuffer = rhi->createBufferDevice(
            "Mesh_ClusterInGroups", size_cast<u32>(meshDataRef->m_meshletInClusterGroup.size() * sizeof(u32)),
            tmpUsage);
        meshResource.cpCounterBuffer =
            rhi->createBufferDevice("Mesh_CpCounter", size_cast<u32>(sizeof(MeshData::GPUCPCounter)), tmpUsage);
        meshResource.tangentBuffer = rhi->createBufferDevice(
            "Mesh_Tangent", size_cast<u32>(sizeof(ifloat4) * meshDataRef->m_tangents.size()), tmpUsage);

        auto materialDataSize = 0;
        auto &materialRef = shaderEffect.m_materials[i];
        if (materialRef->m_data.size() > 0) {
          haveMaterialData = true;
          materialDataSize = size_cast<u32>(materialRef->m_data[0].size());
          meshResource.materialDataBuffer = rhi->createBufferDevice(
              "MaterialData", size_cast<u32>(shaderEffect.m_materials[i]->m_data[0].size()), tmpUsage);
        } else {
          meshResource.materialDataBuffer = nullptr;
          iWarn("Material data not found for mesh {}", i);
        }

        // Indices in bindless descriptors
        meshResource.vertexBufferId = rhi->registerStorageBuffer(meshResource.vertexBuffer.get());
        meshResource.normalBufferId = rhi->registerStorageBuffer(meshResource.normalBuffer.get());
        meshResource.uvBufferId = rhi->registerStorageBuffer(meshResource.uvBuffer.get());
        meshResource.bvhNodeBufferId = rhi->registerStorageBuffer(meshResource.bvhNodeBuffer.get());
        meshResource.clusterGroupBufferId = rhi->registerStorageBuffer(meshResource.clusterGroupBuffer.get());
        meshResource.meshletBufferId = rhi->registerStorageBuffer(meshResource.meshletBuffer.get());
        meshResource.meshletVertexBufferId = rhi->registerStorageBuffer(meshResource.meshletVertexBuffer.get());
        meshResource.meshletIndexBufferId = rhi->registerStorageBuffer(meshResource.meshletIndexBuffer.get());
        meshResource.meshletInClusterBufferId = rhi->registerStorageBuffer(meshResource.meshletInClusterBuffer.get());
        meshResource.cpCounterBufferId = rhi->registerStorageBuffer(meshResource.cpCounterBuffer.get());
        meshResource.tangentBufferId = rhi->registerStorageBuffer(meshResource.tangentBuffer.get());
        if (haveMaterialData) {
          meshResource.materialDataBufferId = rhi->registerStorageBuffer(meshResource.materialDataBuffer.get());
        }

        // Here, we assume that no double bufferring is allowed
        // meaning no CPU-GPU data transfer is allowed for mesh data after
        // initialization
        Mesh::GPUObjectBuffer &objectBuffer = meshResource.objectData;
        objectBuffer.vertexBufferId = meshResource.vertexBufferId->getActiveId();
        objectBuffer.normalBufferId = meshResource.normalBufferId->getActiveId();
        objectBuffer.uvBufferId = meshResource.uvBufferId->getActiveId();
        objectBuffer.bvhNodeBufferId = meshResource.bvhNodeBufferId->getActiveId();
        objectBuffer.clusterGroupBufferId = meshResource.clusterGroupBufferId->getActiveId();
        objectBuffer.meshletBufferId = meshResource.meshletBufferId->getActiveId();
        objectBuffer.meshletVertexBufferId = meshResource.meshletVertexBufferId->getActiveId();
        objectBuffer.meshletIndexBufferId = meshResource.meshletIndexBufferId->getActiveId();
        objectBuffer.meshletInClusterBufferId = meshResource.meshletInClusterBufferId->getActiveId();
        objectBuffer.cpCounterBufferId = meshResource.cpCounterBufferId->getActiveId();
        objectBuffer.boundingSphere = mesh->getBoundingSphere(meshDataRef->m_vertices);
        objectBuffer.materialDataId = haveMaterialData ? meshResource.materialDataBufferId->getActiveId() : 0;
        objectBuffer.tangentBufferId = meshResource.tangentBufferId->getActiveId();

        // description for the whole mesh
        meshResource.objectBuffer = rhi->createBufferDevice("ObjectBuffer", sizeof(Mesh::GPUObjectBuffer), tmpUsage);
        meshResource.objectBufferId = rhi->registerStorageBuffer(meshResource.objectBuffer.get());

        mesh->setGPUResource(meshResource);
      }

      // Setup instance buffers
      auto meshInst = shaderEffect.m_instances[i];
      MeshInstance::GPUResource instanceResource;
      meshInst->getGPUResource(instanceResource);
      auto &meshInstObjData = meshInst->m_resource.objectData;
      if (instanceResource.objectBuffer == nullptr) {

        auto tmpUsage = RhiBufferUsage_CopyDst | RhiBufferUsage_SSBO;
        requireUpdate = true;
        instanceResource.cpQueueBuffer = rhi->createBufferDevice(
            "Render_CpQueue", size_cast<u32>(sizeof(u32) * meshDataRef->m_bvhNodes.size()), tmpUsage);

        auto safeNumMeshlets = meshDataRef->m_numMeshletsEachLod[0];
        if (meshDataRef->m_numMeshletsEachLod.size() > 1)
          safeNumMeshlets += meshDataRef->m_numMeshletsEachLod[1];
        instanceResource.filteredMeshlets =
            rhi->createBufferDevice("Render_FilteredClsters", sizeof(u32) * safeNumMeshlets, tmpUsage);

        instanceResource.cpQueueBufferId = rhi->registerStorageBuffer(instanceResource.cpQueueBuffer.get());
        instanceResource.filteredMeshletsId = rhi->registerStorageBuffer(instanceResource.filteredMeshlets.get());

        instanceResource.objectData.cpQueueBufferId = instanceResource.cpQueueBufferId->getActiveId();
        instanceResource.objectData.filteredMeshletsId = instanceResource.filteredMeshletsId->getActiveId();

        instanceResource.objectBuffer =
            rhi->createBufferDevice("Render_Objects", sizeof(MeshInstance::GPUObjectBuffer), tmpUsage);
        instanceResource.objectBufferId = rhi->registerStorageBuffer(instanceResource.objectBuffer.get());

        meshInst->setGPUResource(instanceResource);
      }

      shaderEffect.m_objectData[i].objectDataRef = meshResource.objectBufferId->getActiveId();
      shaderEffect.m_objectData[i].instanceDataRef = meshInst->m_resource.objectBufferId->getActiveId();
      shaderEffect.m_objectData[i].materialId = curShaderMaterialId;

      // update vertex buffer, TODO: dirty flag
      if (requireUpdate) {
        auto funcEnqueueStagedBuffer = [&](std::shared_ptr<RhiStagedSingleBuffer> stagedBuffer, void *data, u32 size) {
          stagedBuffers.push_back(stagedBuffer);
          pendingVertexBuffers.push_back(data);
          pendingVertexBufferSizes.push_back(size);
        };
#define enqueueStagedBuffer(name, vecBuffer)                                                                           \
  auto staged##name = rhi->createStagedSingleBuffer(meshResource.name.get());                                          \
  funcEnqueueStagedBuffer(                                                                                             \
      staged##name, meshDataRef->vecBuffer.data(),                                                                     \
      size_cast<u32>(meshDataRef->vecBuffer.size() * sizeof(decltype(meshDataRef->vecBuffer)::value_type)))

        enqueueStagedBuffer(vertexBuffer, m_verticesAligned);
        enqueueStagedBuffer(normalBuffer, m_normalsAligned);
        enqueueStagedBuffer(uvBuffer, m_uvs);
        enqueueStagedBuffer(tangentBuffer, m_tangents);
        enqueueStagedBuffer(bvhNodeBuffer, m_bvhNodes);
        enqueueStagedBuffer(clusterGroupBuffer, m_clusterGroups);
        enqueueStagedBuffer(meshletBuffer, m_meshlets);
        enqueueStagedBuffer(meshletVertexBuffer, m_meshletVertices);
        enqueueStagedBuffer(meshletIndexBuffer, m_meshletTriangles);
        enqueueStagedBuffer(meshletInClusterBuffer, m_meshletInClusterGroup);

        auto stagedCPCounterBuffer = rhi->createStagedSingleBuffer(meshResource.cpCounterBuffer.get());
        stagedBuffers.push_back(stagedCPCounterBuffer);
        pendingVertexBuffers.push_back(&meshDataRef->m_cpCounter);
        pendingVertexBufferSizes.push_back(sizeof(MeshData::GPUCPCounter));

        std::shared_ptr<RhiStagedSingleBuffer> stagedMaterialDataBuffer = nullptr;
        if (haveMaterialData) {
          stagedMaterialDataBuffer = rhi->createStagedSingleBuffer(meshResource.materialDataBuffer.get());
          stagedBuffers.push_back(stagedMaterialDataBuffer);
          pendingVertexBuffers.push_back(&shaderEffect.m_materials[i]->m_data[0][0]);
          pendingVertexBufferSizes.push_back(shaderEffect.m_materials[i]->m_data[0].size());
        }

        auto stagedObjectBuffer = rhi->createStagedSingleBuffer(meshResource.objectBuffer.get());
        stagedBuffers.push_back(stagedObjectBuffer);
        pendingVertexBuffers.push_back(&mesh->m_resource.objectData);
        pendingVertexBufferSizes.push_back(sizeof(Mesh::GPUObjectBuffer));

        auto stagedInstanceObjectBuffer = rhi->createStagedSingleBuffer(instanceResource.objectBuffer.get());
        stagedBuffers.push_back(stagedInstanceObjectBuffer);
        pendingVertexBuffers.push_back(&meshInst->m_resource.objectData);
        pendingVertexBufferSizes.push_back(sizeof(MeshInstance::GPUObjectBuffer));

#undef enqueueStagedBuffer
      }
    }

    // update batched object data
    auto batchedObjectData = shaderEffect.m_batchedObjectData;
    auto batchedObjectDataAct = batchedObjectData->getActiveBuffer();
    batchedObjectDataAct->map();
    batchedObjectDataAct->writeBuffer(shaderEffect.m_objectData.data(), sizeof(PerObjectData) * objectCount, 0);
    batchedObjectDataAct->flush();
    batchedObjectDataAct->unmap();

    curShaderMaterialId++;
  }
  // Issue a command buffer to copy data to GPU
  if (stagedBuffers.size() > 0) {
    auto queue = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    queue->runSyncCommand([&](const RhiCommandList *cmd) {
      for (int i = 0; i < stagedBuffers.size(); i++) {
        stagedBuffers[i]->cmdCopyToDevice(cmd, pendingVertexBuffers[i], pendingVertexBufferSizes[i], 0);
      }
    });
  }
}

IFRIT_APIDECL void RendererBase::updateLastFrameTransforms(PerFrameData &perframeData) {
  throw std::runtime_error("Deprecated");
}

IFRIT_APIDECL void RendererBase::endFrame(const std::vector<GPUCommandSubmission *> &cmdToWait) {
  auto rhi = m_app->getRhiLayer();
  using namespace Ifrit::GraphicsBackend;
  auto drawq = rhi->getQueue(Rhi::RhiQueueCapability::RHI_QUEUE_GRAPHICS_BIT);
  auto swapchainImg = rhi->getSwapchainImage();
  auto sRenderComplete = rhi->getSwapchainRenderDoneEventHandler();
  auto cmd = drawq->runAsyncCommand(
      [&](const Rhi::RhiCommandList *cmd) {
        Rhi::RhiTransitionBarrier barrier;
        barrier.m_texture = swapchainImg;
        barrier.m_type = Rhi::RhiResourceType::Texture;
        barrier.m_dstState = Rhi::RhiResourceState::Present;
        barrier.m_subResource = {0, 0, 1, 1};

        Rhi::RhiResourceBarrier barrier2;
        barrier2.m_type = Rhi::RhiBarrierType::Transition;
        barrier2.m_transition = barrier;

        cmd->resourceBarrier({barrier2});
      },
      cmdToWait, {sRenderComplete.get()});
  rhi->endFrame();
}

IFRIT_APIDECL std::unique_ptr<RendererBase::GPUCommandSubmission> RendererBase::beginFrame() {
  auto rhi = m_app->getRhiLayer();
  rhi->beginFrame();
  return rhi->getSwapchainFrameReadyEventHandler();
}

} // namespace Ifrit::Core