
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

#include "ifrit/core/renderer/ayanami/AyanamiSceneAggregator.h"
#include "ifrit/core/renderer/ayanami/AyanamiMeshDF.h"

namespace Ifrit::Core::Ayanami {

struct AyanamiSceneResources {
  using GPUTexture = GraphicsBackend::Rhi::RhiTexture;
  using GPUBuffer = GraphicsBackend::Rhi::RhiBuffer;
  using GPUMultiBuffer = GraphicsBackend::Rhi::RhiMultiBuffer;
  using GPUBindId = GraphicsBackend::Rhi::RhiBindlessIdRef;

  struct MDFDescriptor {
    u32 m_mdfMetaId;
    u32 m_transformId;
  };
  Vec<MDFDescriptor> m_meshMetaIds;

  u32 m_m_mdfAllInstancesAllocSize = 0;
  Ref<GPUMultiBuffer> m_mdfAllInstances = nullptr;
  Ref<GPUBindId> m_mdfAllInstancesBindId;
};

IFRIT_APIDECL void AyanamiSceneAggregator::collectScene(Scene *scene) {
  // Here, filter out all objects with mesh df components

  using Ifrit::Common::Utility::size_cast;

  m_sceneResources->m_meshMetaIds.clear();
  scene->filterObjectsUnsafe([&](SceneObject *obj) {
    auto meshDF = obj->getComponent<AyanamiMeshDF>();
    if (meshDF != nullptr) {
      // Get transform
      auto transform = obj->getComponent<Transform>();
      if (transform == nullptr) {
        iError("AyanamiSceneAggregator::collectScene() requires transform to be attached to a object");
        std::abort();
      }
      // Collect mesh df data
      meshDF->buildGPUResource(m_rhi);
      auto metaId = meshDF->getMetaBufferId();

      AyanamiSceneResources::MDFDescriptor desc;
      desc.m_mdfMetaId = metaId;
      {
        using namespace Ifrit::GraphicsBackend::Rhi;
        Ref<RhiMultiBuffer> transformBuf, transformBufLast;
        Ref<RhiBindlessIdRef> transformBindId, transformBindIdLast;
        transform->getGPUResource(transformBuf, transformBufLast, transformBindId, transformBindIdLast);
        desc.m_transformId = transformBindId->getActiveId();
      }
      m_sceneResources->m_meshMetaIds.push_back(desc);
    }
    return false;
  });

  // All instances gathered, now build the all instance buffer
  if (m_sceneResources->m_mdfAllInstances != nullptr &&
      m_sceneResources->m_m_mdfAllInstancesAllocSize != m_sceneResources->m_meshMetaIds.size()) {
    iError("AyanamiSceneAggregator::collectScene() does not support dynamic scene now");
    std::abort();
  }

  if (m_sceneResources->m_mdfAllInstances == nullptr) {
    m_sceneResources->m_m_mdfAllInstancesAllocSize = m_sceneResources->m_meshMetaIds.size();
    m_sceneResources->m_mdfAllInstances = m_rhi->createBufferCoherent(
        sizeof(AyanamiSceneResources::MDFDescriptor) * m_sceneResources->m_m_mdfAllInstancesAllocSize,
        GraphicsBackend::Rhi::RhiBufferUsage::RhiBufferUsage_CopyDst ||
            GraphicsBackend::Rhi::RhiBufferUsage::RhiBufferUsage_SSBO);
  }

  // update the buffer
  auto activeBuf = m_sceneResources->m_mdfAllInstances->getActiveBuffer();
  activeBuf->map();
  activeBuf->writeBuffer(
      m_sceneResources->m_meshMetaIds.data(),
      size_cast<u32>(m_sceneResources->m_meshMetaIds.size() * sizeof(AyanamiSceneResources::MDFDescriptor)), 0);
  activeBuf->flush();
  activeBuf->unmap();
}

IFRIT_APIDECL u32 AyanamiSceneAggregator::getGatheredBufferId() {
  return m_sceneResources->m_mdfAllInstancesBindId->getActiveId();
}

IFRIT_APIDECL void AyanamiSceneAggregator::init() { m_sceneResources = new AyanamiSceneResources(); }

IFRIT_APIDECL void AyanamiSceneAggregator::destroy() {
  if (m_sceneResources != nullptr) {
    delete m_sceneResources;
    m_sceneResources = nullptr;
  }
}

} // namespace Ifrit::Core::Ayanami