
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
  float m_renderWidthf;
  float m_renderHeightf;
  float m_cameraNear;
  float m_cameraFar;
  float m_cameraFovX;
  float m_cameraFovY;
  float m_cameraAspect;
  float m_cameraOrthoSize;
  float m_hizLods;
  float m_viewCameraType; // 0: perspective, 1: ortho

  float m_cullCamOrthoSizeX;
  float m_cullCamOrthoSizeY;
};

struct PerObjectData {
  uint32_t transformRef = 0;
  uint32_t objectDataRef = 0;
  uint32_t instanceDataRef = 0;
  uint32_t transformRefLast = 0;
  uint32_t materialId = 0;
};

struct PerShaderEffectData {
  std::vector<Material *> m_materials;
  std::vector<Mesh *> m_meshes;
  std::vector<Transform *> m_transforms;
  std::vector<MeshInstance *> m_instances;

  // Data to GPUs
  uint32_t m_lastObjectCount = ~0u;
  std::vector<PerObjectData> m_objectData;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiMultiBuffer>
      m_batchedObjectData = nullptr;
  Ifrit::GraphicsBackend::Rhi::RhiBindlessDescriptorRef *m_batchedObjBufRef =
      nullptr;
};

struct PerFrameRenderTargets {
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiTexture> m_colorRT;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef> m_colorRTId;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef> m_colorRTIdSRV;
  Ifrit::GraphicsBackend::Rhi::RhiTexture *m_depthRT;

  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiColorAttachment> m_colorRTRef;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiDepthStencilAttachment>
      m_depthRTRef;
  std::shared_ptr<Ifrit::GraphicsBackend::Rhi::RhiRenderTargets> m_rts;
  uint32_t m_width = 0, m_height = 0;
};

struct ShadowMappingData {
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;
  using GPUUniformBuffer = Ifrit::GraphicsBackend::Rhi::RhiMultiBuffer;
  using GPUBindlessId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;

  struct SingleShadowView {
    std::array<uint32_t, 4> m_viewRef;
    std::array<uint32_t, 4> m_texRef;

    std::array<uint32_t, 4> m_viewMapping;
    std::array<float, 4> m_csmStart;
    std::array<float, 4> m_csmEnd;
    uint32_t m_csmSplits;
  };
  std::vector<SingleShadowView> m_shadowViews;
  uint32_t m_enabledShadowMaps = 0;

  std::shared_ptr<GPUUniformBuffer> m_allShadowData = nullptr;
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
    std::shared_ptr<GPUTexture> m_specular_occlusion_intermediate;
    std::shared_ptr<GPUTexture> m_normal_smoothness;
    std::shared_ptr<GPUTexture> m_emissive;
    std::shared_ptr<GPUTexture> m_shadowMask;

    uint32_t m_rtWidth = 0;
    uint32_t m_rtHeight = 0;
    uint32_t m_rtCreated = 0;

    std::shared_ptr<GPUBindlessId> m_albedo_materialFlagsId;
    std::shared_ptr<GPUBindlessId> m_albedo_materialFlags_sampId;
    std::shared_ptr<GPUBindlessId> m_specular_occlusionId;
    std::shared_ptr<GPUBindlessId> m_specular_occlusion_sampId;
    std::shared_ptr<GPUBindlessId> m_specular_occlusion_intermediateId;
    std::shared_ptr<GPUBindlessId> m_specular_occlusion_intermediate_sampId;
    std::shared_ptr<GPUBindlessId> m_normal_smoothnessId;
    std::shared_ptr<GPUBindlessId> m_normal_smoothness_sampId;
    std::shared_ptr<GPUBindlessId> m_emissiveId;
    std::shared_ptr<GPUBindlessId> m_shadowMaskId;

    std::shared_ptr<GPUBindlessId> m_depth_sampId;

    GPUBarrier m_normal_smoothnessBarrier;
    GPUBarrier m_specular_occlusionBarrier;

    std::shared_ptr<GPUBuffer> m_gbufferRefs = nullptr;
    GPUBindlessRef *m_gbufferDesc = nullptr;

    std::vector<GPUBarrier> m_gbufferBarrier;
  };

  struct SinglePassHiZData {
    std::shared_ptr<GPUTexture> m_hizTexture = nullptr;
    std::vector<uint32_t> m_hizRefs;
    std::shared_ptr<GPUBuffer> m_hizRefBuffer = nullptr;
    std::shared_ptr<GPUBindlessId> m_hizRefBufferId = nullptr;
    std::shared_ptr<GPUBuffer> m_hizAtomics = nullptr;
    GPUBindlessRef *m_hizDesc = nullptr;
    uint32_t m_hizIters = 0;
    uint32_t m_hizWidth = 0;
    uint32_t m_hizHeight = 0;
  };

  struct PerViewData {
    ViewType m_viewType = ViewType::Invisible;

    PerFramePerViewData m_viewData;
    PerFramePerViewData m_viewDataOld;
    std::shared_ptr<GPUUniformBuffer> m_viewBuffer = nullptr;
    std::shared_ptr<GPUUniformBuffer> m_viewBufferLast = nullptr;
    GPUBindlessRef *m_viewBindlessRef = nullptr;
    std::shared_ptr<GPUBindlessId> m_viewBufferId = nullptr;

    // Non-gpu data
    uint32_t m_renderWidth;
    uint32_t m_renderHeight;
    bool m_camMoved;

    // visibility buffer
    std::shared_ptr<GPUTexture> m_visibilityBuffer_HW = nullptr;
    std::shared_ptr<GPUTexture> m_visPassDepth_HW = nullptr;
    std::shared_ptr<GPUBindlessId> m_visBufferIdUAV_HW = nullptr;

    std::shared_ptr<GPUColorRT> m_visColorRT_HW = nullptr;
    std::shared_ptr<GPUDepthRT> m_visDepthRT_HW = nullptr;
    std::shared_ptr<GPURTs> m_visRTs_HW = nullptr;
    std::shared_ptr<GPUBindlessId> m_visDepthId_HW = nullptr;

    // visibility buffer software. It's compute shader, so
    // not repeated decl required
    std::shared_ptr<GPUTexture> m_visibilityBuffer_SW = nullptr;
    std::shared_ptr<GPUBuffer> m_visPassDepth_SW = nullptr;
    std::shared_ptr<GPUBuffer> m_visPassDepthCASLock_SW = nullptr;

    std::shared_ptr<GPUBindlessId> m_visBufferIdUAV_SW = nullptr;
    std::shared_ptr<GPUBindlessId> m_visDepthId_SW = nullptr;
    std::shared_ptr<GPUBindlessId> m_visDepthCASLockId_SW = nullptr;

    // combined visibility buffer is required
    std::shared_ptr<GPUTexture> m_visibilityBuffer_Combined = nullptr;
    std::shared_ptr<GPUTexture> m_visibilityDepth_Combined = nullptr;

    std::shared_ptr<GPUBindlessId> m_visibilityBufferIdUAV_Combined = nullptr;
    std::shared_ptr<GPUBindlessId> m_visibilityDepthIdUAV_Combined = nullptr;
    std::shared_ptr<GPUBindlessId> m_visibilityBufferIdSRV_Combined = nullptr;
    std::shared_ptr<GPUBindlessId> m_visibilityDepthIdSRV_Combined = nullptr;

    // visibility buffer for 2nd pass, reference to the same texture, but
    // without clearing
    std::shared_ptr<GPUColorRT> m_visColorRT2_HW = nullptr;
    std::shared_ptr<GPUDepthRT> m_visDepthRT2_HW = nullptr;
    std::shared_ptr<GPURTs> m_visRTs2_HW = nullptr;

    // all visible clusters
    std::shared_ptr<GPUBuffer> m_allFilteredMeshletsAllCount = nullptr;

    std::shared_ptr<GPUBuffer> m_allFilteredMeshletsHW = nullptr;
    std::shared_ptr<GPUBuffer> m_allFilteredMeshletsSW = nullptr;

    uint32_t m_allFilteredMeshlets_SWOffset = 0;

    uint32_t m_allFilteredMeshletsMaxCount = 0;
    uint32_t m_requireMaxFilteredMeshlets = 0;
    GPUBindlessRef *m_allFilteredMeshletsDesc = nullptr;

    // SPD HiZ
    SinglePassHiZData m_spHiZData;
    SinglePassHiZData m_spHiZDataMin;

    // Instance culling
    std::shared_ptr<GPUBuffer> m_instCullDiscardObj = nullptr;
    std::shared_ptr<GPUBuffer> m_instCullPassedObj = nullptr;
    std::shared_ptr<GPUBuffer> m_persistCullIndirectDispatch = nullptr;
    GPUBindlessRef *m_instCullDesc = nullptr;
    uint32_t m_maxSupportedInstances = 0;

    // Inst-Persist barrier
    std::vector<GPUBarrier> m_persistCullBarrier;
    std::vector<GPUBarrier> m_visibilityBarrier;
  };

  struct FSR2ExtraData {
    std::shared_ptr<GPUTexture> m_fsr2Output = nullptr;
    std::shared_ptr<GPUBindlessId> m_fsr2OutputSRVId = nullptr;
    uint32_t m_fsrFrameId = 0;
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

  // Gbuffer desc
  GPUBindlessRef *m_gbufferDescFrag = nullptr;

  // Emit depth targets
  std::shared_ptr<GPUTexture> m_velocityMaterial = nullptr;
  std::shared_ptr<GPUTexture> m_motionVector = nullptr;
  GPUBindlessRef *m_velocityMaterialDesc = nullptr;

  // Material classify
  std::shared_ptr<GPUBuffer> m_matClassCountBuffer = nullptr;
  std::shared_ptr<GPUBuffer> m_matClassIndirectDispatchBuffer = nullptr;
  std::shared_ptr<GPUBuffer> m_matClassFinalBuffer = nullptr;
  std::shared_ptr<GPUBuffer> m_matClassPixelOffsetBuffer = nullptr;
  GPUBindlessRef *m_matClassDesc = nullptr;
  std::vector<GPUBarrier> m_matClassBarrier;
  uint32_t m_matClassSupportedNumMaterials = 0;
  uint32_t m_matClassSupportedNumPixels = 0;

  // For history
  uint32_t m_frameId = 0;
  float m_frameTimestamp[2] = {0.0f, 0.0f};

  // TAA
  std::vector<PerFrameRenderTargets> m_taaHistory;
  std::shared_ptr<GPUTexture> m_taaUnresolved = nullptr;
  GPUBindlessRef *m_taaHistoryDesc = nullptr;
  float m_taaJitterX = 0.0f;
  float m_taaJitterY = 0.0f;

  // FSR2
  FSR2ExtraData m_fsr2Data;

  // Atmosphere
  std::shared_ptr<void> m_atmosphereData = nullptr;
  std::shared_ptr<GPUTexture> m_atmoOutput;
  std::shared_ptr<GPUBindlessId> m_atmoOutputId;
  ifloat4 m_sunDir;

  // Shadow mapping
  ShadowMappingData m_shadowData2;
  std::shared_ptr<GPUTexture> m_deferShadowMask = nullptr;
  std::shared_ptr<GPUColorRT> m_deferShadowMaskRT;
  std::shared_ptr<GPURTs> m_deferShadowMaskRTs;

  std::shared_ptr<GPUBindlessId> m_deferShadowMaskId;
};

} // namespace Ifrit::Core