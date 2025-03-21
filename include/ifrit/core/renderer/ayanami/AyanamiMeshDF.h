
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

#include "ifrit/core/base/Component.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core::Ayanami {

struct AyanamiMeshDFResource {
  struct SDFMeta {
    ifloat4 bboxMin;
    ifloat4 bboxMax;
    u32 width;
    u32 height;
    u32 depth;
    u32 sdfId;
  };
  using GPUTexture = GraphicsBackend::Rhi::RhiTexture;
  using GPUSampler = GraphicsBackend::Rhi::RhiSampler;
  using GPUBuffer = GraphicsBackend::Rhi::RhiBuffer;
  using GPUBindId = GraphicsBackend::Rhi::RhiBindlessIdRef;

  Ref<GPUTexture> sdfTexture;
  Ref<GPUBindId> sdfTextureBindId;
  Ref<GPUBuffer> sdfMetaBuffer;
  Ref<GPUBindId> sdfMetaBufferBindId;
  Ref<GPUSampler> sdfSampler; // this design is not a good idea, should be removed in the future
};

// AyanamiMeshDF stores mesh-level signed distance field data
// This is used for mesh-based raymarching
class IFRIT_APIDECL AyanamiMeshDF : public Component {
private:
  Vec<f32> m_sdfData;
  u32 m_sdWidth;
  u32 m_sdHeight;
  u32 m_sdDepth;
  ifloat3 m_sdBoxMin;
  ifloat3 m_sdBoxMax;
  bool m_isBuilt = false;

  Uref<AyanamiMeshDFResource> m_gpuResource = nullptr;

public:
  AyanamiMeshDF() {}
  AyanamiMeshDF(std::shared_ptr<SceneObject> owner) : Component(owner) {}
  virtual ~AyanamiMeshDF() = default;

  inline std::string serialize() override { return ""; }
  inline void deserialize() override {}

public:
  void buildMeshDF(const std::string_view &cachePath);
  void buildGPUResource(GraphicsBackend::Rhi::RhiBackend *rhi);
  inline u32 getMetaBufferId() const { return m_gpuResource->sdfMetaBufferBindId->getActiveId(); }
  IFRIT_COMPONENT_SERIALIZE(m_sdWidth, m_sdHeight, m_sdDepth, m_sdBoxMin, m_sdBoxMax);
};

} // namespace Ifrit::Core::Ayanami

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Ayanami::AyanamiMeshDF)
