
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

#pragma once
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/core/base/Material.h"
#include "ifrit/core/base/Mesh.h"
#include "ifrit/core/base/Object.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <unordered_map>
#include <vector>

namespace Ifrit::Core {

// The design is inappropriate for the current project.
// It's renderer-specific and should be moved to the renderer.

struct PerFramePerViewData {
  float4x4 m_worldToView;
  float4x4 m_perspective;
  float4x4 m_worldToClip;
  float4x4 m_inversePerspective;
  float4x4 m_clipToWorld;
  float4x4 m_viewToWorld;
  ifloat4 m_cameraPosition;
  ifloat4 m_cameraFront;
  float m_renderWidth;
  float m_renderHeight;
  float m_cameraNear;
  float m_cameraFar;
  float m_cameraFovX;
  float m_cameraFovY;
  float m_cameraAspect;
  float m_cameraOrthoSize;
  float m_hizLods;
  float m_viewCameraType; // 0: perspective, 1: ortho
};

struct PerObjectData {
  uint32_t transformRef = 0;
  uint32_t objectDataRef = 0;
  uint32_t instanceDataRef = 0;
  uint32_t transformRefLast = 0;
  uint32_t materialId = 0;
};

struct PerShaderEffectData {
  std::vector<std::shared_ptr<Material>> m_materials;
  std::vector<std::shared_ptr<Mesh>> m_meshes;
  std::vector<std::shared_ptr<Transform>> m_transforms;
  std::vector<std::shared_ptr<MeshInstance>> m_instances;

  // Data to GPUs
  uint32_t m_lastObjectCount = ~0u;
  std::vector<PerObjectData> m_objectData;
  Ifrit::GraphicsBackend::Rhi::RhiMultiBuffer *m_batchedObjectData = nullptr;
  Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef *m_batchedObjBufRef =
      nullptr;
};

struct PerFrameRenderTargets {
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiTexture> m_colorRT;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef> m_colorRTId;
  Ifrit::GraphicsBackend::Rhi::RhiTexture *m_depthRT;

  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiColorAttachment> m_colorRTRef;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiDepthStencilAttachment>
      m_depthRTRef;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiRenderTargets> m_rts;
  uint32_t m_width = 0, m_height = 0;
};

struct ShadowMappingData {
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;
  using GPUBindlessId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;

  struct SingleShadowView {
    uint32_t m_viewRef;
    uint32_t m_texRef;
  };
  std::vector<SingleShadowView> m_shadowViews;
  GPUBuffer *m_allShadowData;
  std::shared_ptr<GPUBindlessId> m_allShadowDataId;
};

struct PerFrameData {
  using GPUUniformBuffer = Ifrit::GraphicsBackend::Rhi::RhiMultiBuffer;
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;
  using GPUBindlessRef = Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef;
  using GPUBindlessId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUColorRT = Ifrit::GraphicsBackend::Rhi::RhiColorAttachment;
  using GPUDepthRT = Ifrit::GraphicsBackend::Rhi::RhiDepthStencilAttachment;
  using GPURTs = Ifrit::GraphicsBackend::Rhi::RhiRenderTargets;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSampler;
  using GPUBarrier = Ifrit::GraphicsBackend::Rhi::RhiResourceBarrier;

  enum class ViewType { Invisible, Primary, Display, Shadow };

  struct GBufferDesc {
    uint32_t m_albedo_materialFlags;
    uint32_t m_specular_occlusion;
    uint32_t m_normal_smoothness;
    uint32_t m_emissive;
    uint32_t m_shadowMask;
  };

  struct GBuffer {
    std::shared_ptr<GPUTexture> m_albedo_materialFlags;
    std::shared_ptr<GPUTexture> m_specular_occlusion;
    std::shared_ptr<GPUTexture> m_normal_smoothness;
    std::shared_ptr<GPUTexture> m_emissive;
    std::shared_ptr<GPUTexture> m_shadowMask;

    uint32_t m_rtWidth = 0;
    uint32_t m_rtHeight = 0;
    uint32_t m_rtCreated = 0;

    std::shared_ptr<GPUSampler> m_gbufferSampler = nullptr;

    std::shared_ptr<GPUBindlessId> m_albedo_materialFlagsId;
    std::shared_ptr<GPUBindlessId> m_specular_occlusionId;
    std::shared_ptr<GPUBindlessId> m_specular_occlusion_sampId;
    std::shared_ptr<GPUBindlessId> m_normal_smoothnessId;
    std::shared_ptr<GPUBindlessId> m_normal_smoothness_sampId;
    std::shared_ptr<GPUBindlessId> m_emissiveId;
    std::shared_ptr<GPUBindlessId> m_shadowMaskId;

    std::shared_ptr<GPUBindlessId> m_depth_sampId;

    GPUBarrier m_normal_smoothnessBarrier;
    GPUBarrier m_specular_occlusionBarrier;

    GPUBuffer *m_gbufferRefs = nullptr;
    GPUBindlessRef *m_gbufferDesc = nullptr;

    std::vector<GPUBarrier> m_gbufferBarrier;
  };

  struct SinglePassHiZData {
    std::shared_ptr<GPUTexture> m_hizTexture = nullptr;
    std::vector<uint32_t> m_hizRefs;
    GPUBuffer *m_hizRefBuffer = nullptr;
    GPUBuffer *m_hizAtomics = nullptr;
    GPUBindlessRef *m_hizDesc = nullptr;
    std::shared_ptr<GPUSampler> m_hizSampler = nullptr;
    uint32_t m_hizIters = 0;
    uint32_t m_hizWidth = 0;
    uint32_t m_hizHeight = 0;
  };

  struct PerViewData {
    ViewType m_viewType = ViewType::Invisible;

    PerFramePerViewData m_viewData;
    PerFramePerViewData m_viewDataOld;
    GPUUniformBuffer *m_viewBuffer = nullptr;
    GPUUniformBuffer *m_viewBufferLast = nullptr;
    GPUBindlessRef *m_viewBindlessRef = nullptr;
    std::shared_ptr<GPUBindlessId> m_viewBufferId = nullptr;

    // visibility buffer
    std::shared_ptr<GPUTexture> m_visibilityBuffer = nullptr;
    GPUTexture *m_visPassDepth = nullptr;
    std::shared_ptr<GPUSampler> m_visDepthSampler = nullptr;

    std::shared_ptr<GPUColorRT> m_visColorRT = nullptr;
    std::shared_ptr<GPUDepthRT> m_visDepthRT = nullptr;
    std::shared_ptr<GPURTs> m_visRTs = nullptr;
    std::shared_ptr<GPUBindlessId> m_visDepthId = nullptr;

    // visibility buffer for 2nd pass, reference to the same texture, but
    // without clearing
    std::shared_ptr<GPUColorRT> m_visColorRT2 = nullptr;
    std::shared_ptr<GPUDepthRT> m_visDepthRT2 = nullptr;
    std::shared_ptr<GPURTs> m_visRTs2 = nullptr;

    // all visible clusters
    GPUBuffer *m_allFilteredMeshlets = nullptr;
    GPUBuffer *m_allFilteredMeshletsCount = nullptr;
    uint32_t m_allFilteredMeshletsMaxCount = 0;
    uint32_t m_requireMaxFilteredMeshlets = 0;
    GPUBindlessRef *m_allFilteredMeshletsDesc = nullptr;

    // Hiz buffer
    std::shared_ptr<GPUTexture> m_hizTexture = nullptr;
    std::vector<GPUBindlessRef *> m_hizDescs;
    std::vector<GPUUniformBuffer *> m_hizTexSize;
    std::shared_ptr<GPUSampler> m_hizDepthSampler = nullptr;
    uint32_t m_hizIter = 0;

    std::vector<std::shared_ptr<GPUBindlessId>> m_hizTestMips;
    std::vector<uint32_t> m_hizTestMipsId;
    GPUBuffer *m_hizTestMipsBuffer = nullptr;
    GPUBindlessRef *m_hizTestDesc = nullptr;

    // SPD HiZ
    SinglePassHiZData m_spHiZData;

    // Instance culling
    GPUBuffer *m_instCullDiscardObj = nullptr;
    GPUBuffer *m_instCullPassedObj = nullptr;
    GPUBuffer *m_persistCullIndirectDispatch = nullptr;
    GPUBindlessRef *m_instCullDesc = nullptr;
    uint32_t m_maxSupportedInstances = 0;

    // Inst-Persist barrier
    std::vector<GPUBarrier> m_persistCullBarrier;
    std::vector<GPUBarrier> m_visibilityBarrier;
  };
  constexpr static Ifrit::GraphicsBackend::Rhi::RhiImageFormat
      c_visibilityFormat =
          Ifrit::GraphicsBackend::Rhi::RhiImageFormat::RHI_FORMAT_R32_UINT;

  std::unordered_set<uint32_t> m_enabledEffects;
  std::vector<PerShaderEffectData> m_shaderEffectData;
  std::unordered_map<ShaderEffect, uint32_t, ShaderEffectHash>
      m_shaderEffectMap;
  PerShaderEffectData m_allInstanceData;

  // Per view data. Here for simplicity, assume 0 is the primary view
  // Other views are for shadow maps, etc.
  std::vector<PerViewData> m_views;

  // GBuffer
  GBuffer m_gbuffer;
  GPUBindlessRef *m_gbufferDepthDesc;
  std::shared_ptr<GPUSampler> m_gbufferDepthSampler = nullptr;
  std::shared_ptr<GPUBindlessId> m_gbufferDepthIdX = nullptr;

  // Gbuffer desc
  std::shared_ptr<GPUSampler> m_gbufferSampler = nullptr;
  GPUBindlessRef *m_gbufferDescFrag = nullptr;

  // Visibility show
  std::shared_ptr<GPUSampler> m_visibilitySampler = nullptr;
  GPUBindlessRef *m_visShowCombinedRef = nullptr;

  // Emit depth targets
  std::shared_ptr<GPUTexture> m_velocityMaterial = nullptr;
  GPUBindlessRef *m_velocityMaterialDesc = nullptr;

  // Material classify
  GPUBuffer *m_matClassCountBuffer = nullptr;
  GPUBuffer *m_matClassIndirectDispatchBuffer = nullptr;
  GPUBuffer *m_matClassFinalBuffer = nullptr;
  GPUBuffer *m_matClassPixelOffsetBuffer = nullptr;
  GPUBindlessRef *m_matClassDesc = nullptr;
  std::vector<GPUBarrier> m_matClassBarrier;
  std::shared_ptr<GPUTexture> m_matClassDebug = nullptr;
  uint32_t m_matClassSupportedNumMaterials = 0;
  uint32_t m_matClassSupportedNumPixels = 0;

  // For history
  uint32_t m_frameId = 0;

  // TAA
  std::vector<PerFrameRenderTargets> m_taaHistory;
  std::shared_ptr<GPUTexture> m_taaUnresolved = nullptr;
  std::shared_ptr<GPUSampler> m_taaHistorySampler = nullptr;
  GPUBindlessRef *m_taaHistoryDesc = nullptr;
  std::shared_ptr<GPUSampler> m_taaSampler = nullptr;
  float m_taaJitterX = 0.0f;
  float m_taaJitterY = 0.0f;

  // Atmosphere
  std::shared_ptr<void> m_atmosphereData = nullptr;
  std::shared_ptr<GPUTexture> m_atmoOutput;
  std::shared_ptr<GPUBindlessId> m_atmoOutputId;

  // Shadow mapping
  ShadowMappingData m_shadowData;
};

} // namespace Ifrit::Core