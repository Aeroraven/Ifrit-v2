
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
  }
}

} // namespace Ifrit::Core::Ayanami