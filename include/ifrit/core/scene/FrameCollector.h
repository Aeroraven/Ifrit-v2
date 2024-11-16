#pragma once
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/core/base/Material.h"
#include "ifrit/core/base/Mesh.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <unordered_map>
#include <vector>

namespace Ifrit::Core {

struct PerFramePerViewData {
  float4x4 m_worldToView;
  float4x4 m_perspective;
  ifloat4 m_cameraPosition;
  float m_renderWidth;
  float m_renderHeight;
  float m_cameraNear;
  float m_cameraFar;
  float m_cameraFovX;
  float m_cameraFovY;
  float m_cameraAspect;
  float m_hizLods;
};

struct PerObjectData {
  uint32_t transformRef = 0;
  uint32_t objectDataRef = 0;
  uint32_t instanceDataRef = 0;
  uint32_t transformRefLast = 0;
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

  PerFramePerViewData m_viewData;
  PerFramePerViewData m_viewDataOld;
  GPUUniformBuffer *m_viewBuffer = nullptr;
  GPUUniformBuffer *m_viewBufferLast = nullptr;
  GPUBindlessRef *m_viewBindlessRef = nullptr;
  std::vector<PerShaderEffectData> m_shaderEffectData;
  std::unordered_map<ShaderEffect, uint32_t, ShaderEffectHash>
      m_shaderEffectMap;

  // For culling
  PerShaderEffectData m_allInstanceData;

  // TODO: resource release
  std::unordered_set<uint32_t> m_enabledEffects;

  // visibility buffer
  std::shared_ptr<GPUTexture> m_visibilityBuffer = nullptr;
  GPUTexture *m_visPassDepth = nullptr;
  constexpr static Ifrit::GraphicsBackend::Rhi::RhiImageFormat
      c_visibilityFormat =
          Ifrit::GraphicsBackend::Rhi::RhiImageFormat::RHI_FORMAT_R32_UINT;
  std::shared_ptr<GPUColorRT> m_visColorRT = nullptr;
  std::shared_ptr<GPUDepthRT> m_visDepthRT = nullptr;
  std::shared_ptr<GPURTs> m_visRTs = nullptr;

  // visibility buffer for 2nd pass, reference to the same texture, but without
  // clearing
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

  // Visibility show
  std::shared_ptr<GPUSampler> m_visibilitySampler = nullptr;
  GPUBindlessRef *m_visShowCombinedRef = nullptr;
};

} // namespace Ifrit::Core