
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

#include "ifrit/core/renderer/ayanami/AyanamiMeshDF.h"
#include "ifrit/core/base/Mesh.h"

#define IFRIT_MESHPROC_IMPORT
#include "ifrit/meshproc/engine/base/MeshDesc.h"
#include "ifrit/meshproc/engine/meshsdf/MeshSDFConverter.h"
#undef IFRIT_MESHPROC_IMPORT

#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include <filesystem>

namespace Ifrit::Core::Ayanami {

IF_CONSTEXPR u32 cAyanamiMeshDFWidth = 64;

IFRIT_APIDECL void AyanamiMeshDF::buildMeshDF(const std::string_view &cachePath) {
  auto meshFilter = this->getParentUnsafe()->getComponentUnsafe<MeshFilter>();
  if (meshFilter == nullptr) {
    iError("AyanamiMeshDF::buildMeshDF() requires mesh to be attached to a object");
    std::abort();
  }
  auto meshContainer = meshFilter->getMesh();
  auto meshData = meshContainer->loadMesh();

  {
    using namespace Ifrit::MeshProcLib::MeshSDFProcess;
    using namespace Ifrit::MeshProcLib;
    using namespace Ifrit::Common::Utility;
    MeshDescriptor meshDesc;
    meshDesc.indexCount = size_cast<int>(meshData->m_indices.size());
    meshDesc.indexData = reinterpret_cast<i8 *>(meshData->m_indices.data());
    meshDesc.positionOffset = 0;
    meshDesc.vertexCount = size_cast<int>(meshData->m_vertices.size());
    meshDesc.vertexData = reinterpret_cast<i8 *>(meshData->m_vertices.data());
    meshDesc.vertexStride = sizeof(ifloat3);
    meshDesc.normalData = reinterpret_cast<i8 *>(meshData->m_normals.data());
    meshDesc.normalStride = sizeof(ifloat3);

    auto serialMeshDFName = "core.ayanami.meshdf_1." + meshData->identifier + ".cache";
    bool hasCachedDF = false;
    bool shouldGenCachedDF = false;
    auto cachePathStr = std::string(cachePath);

    if (!std::filesystem::exists(cachePathStr)) {
      shouldGenCachedDF = true;
    } else {
      auto serialMeshDFPath = cachePathStr + serialMeshDFName;
      hasCachedDF = std::filesystem::exists(serialMeshDFPath);
      if (hasCachedDF) {
        shouldGenCachedDF = false;
      } else {
        shouldGenCachedDF = true;
      }
    }

    SignedDistanceField sdf;
    if (hasCachedDF) {
      auto serialMeshDFPath = cachePathStr + serialMeshDFName;
      auto buffer = Ifrit::Common::Utility::readBinaryFile(serialMeshDFPath);
      Ifrit::Common::Serialization::deserializeBinary(buffer, sdf);
    } else {
      iInfo("Building mesh distance field for {}", meshData->identifier);
      convertMeshToSDF(meshDesc, sdf, cAyanamiMeshDFWidth, cAyanamiMeshDFWidth, cAyanamiMeshDFWidth);
      auto serialMeshDFPath = cachePathStr + serialMeshDFName;
      if (shouldGenCachedDF) {
        std::string buffer;
        Ifrit::Common::Serialization::serializeBinary(sdf, buffer);
        Ifrit::Common::Utility::writeBinaryFile(serialMeshDFPath, buffer);
      }
    }
    m_sdfData = std::move(sdf.sdfData);
    m_sdWidth = sdf.width;
    m_sdHeight = sdf.height;
    m_sdDepth = sdf.depth;
    m_sdBoxMin = ifloat3(sdf.bboxMin.x, sdf.bboxMin.y, sdf.bboxMin.z);
    m_sdBoxMax = ifloat3(sdf.bboxMax.x, sdf.bboxMax.y, sdf.bboxMax.z);
    m_isBuilt = true;
  }
}

IFRIT_APIDECL void AyanamiMeshDF::buildGPUResource(GraphicsBackend::Rhi::RhiBackend *rhi) {
  if (m_gpuResource == nullptr) {
    if (m_isBuilt == false) {
      iError("AyanamiMeshDF::buildGPUResource() requires mesh to be built first");
      std::abort();
    }
    m_gpuResource = std::make_unique<AyanamiMeshDFResource>();
    using namespace Ifrit::GraphicsBackend::Rhi;
    auto volumeSize = m_sdWidth * m_sdHeight * m_sdDepth;
    auto deviceVolume =
        rhi->createBuffer(volumeSize * sizeof(f32),
                          RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_CopySrc, true);
    deviceVolume->map();
    deviceVolume->writeBuffer(m_sdfData.data(), volumeSize * sizeof(f32), 0);
    deviceVolume->flush();
    deviceVolume->unmap();
    m_gpuResource->sdfTexture = rhi->createTexture3D(
        m_sdWidth, m_sdHeight, m_sdDepth, RhiImageFormat::RHI_FORMAT_R32_SFLOAT,
        RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT | RhiImageUsage::RHI_IMAGE_USAGE_TRANSFER_DST_BIT);
    m_gpuResource->sdfTextureBindId = rhi->registerUAVImage(m_gpuResource->sdfTexture.get(), {0, 0, 1, 1});
    m_gpuResource->sdfMetaBuffer =
        rhi->createBuffer(sizeof(AyanamiMeshDFResource::SDFMeta),
                          RhiBufferUsage::RhiBufferUsage_CopyDst | RhiBufferUsage::RhiBufferUsage_SSBO, true);
    m_gpuResource->sdfMetaBufferBindId = rhi->registerStorageBuffer(m_gpuResource->sdfMetaBuffer.get());

    auto stagedMetaBuffer = rhi->createStagedSingleBuffer(m_gpuResource->sdfMetaBuffer.get());
    AyanamiMeshDFResource::SDFMeta sdfMeta;
    sdfMeta.bboxMin = ifloat4(m_sdBoxMin.x, m_sdBoxMin.y, m_sdBoxMin.z, 0);
    sdfMeta.bboxMax = ifloat4(m_sdBoxMax.x, m_sdBoxMax.y, m_sdBoxMax.z, 0);
    sdfMeta.width = m_sdWidth;
    sdfMeta.height = m_sdHeight;
    sdfMeta.depth = m_sdDepth;
    sdfMeta.sdfId = m_gpuResource->sdfTextureBindId->getActiveId();

    auto tq = rhi->getQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
    tq->runSyncCommand([&](const RhiCommandBuffer *cmd) {
      cmd->copyBufferToImage(deviceVolume.get(), m_gpuResource->sdfTexture.get(), {0, 0, 1, 1});
      stagedMetaBuffer->cmdCopyToDevice(cmd, &sdfMeta, sizeof(AyanamiMeshDFResource::SDFMeta), 0);
    });
  }
}

} // namespace Ifrit::Core::Ayanami