
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

#include "ifrit/core/base/Mesh.h"
#define IFRIT_MESHPROC_IMPORT
#include "ifrit/meshproc/engine/mesh/MeshClusterLodProc.h"
#include "ifrit/meshproc/engine/mesh/MeshletConeCull.h"

#include "ifrit/meshproc/engine/base/MeshDesc.h"

#undef IFRIT_MESHPROC_IMPORT
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/common/util/FileOps.h"
#include <filesystem>

using namespace Ifrit::Math::SIMD;

namespace Ifrit::Core {

struct CreateMeshLodHierMiscInfo {
  u32 totalLods;

  IFRIT_STRUCT_SERIALIZE(totalLods);
};

IFRIT_APIDECL void Mesh::createMeshLodHierarchy(std::shared_ptr<MeshData> meshData, const String &cachePath) {
  using namespace Ifrit::MeshProcLib;
  using namespace Ifrit::MeshProcLib::MeshProcess;
  using namespace Ifrit::Common::Utility;
  const size_t max_vertices = 64;
  const size_t max_triangles = 124;
  const float cone_weight = 0.0f;
  constexpr int MAX_LOD = 10;

  MeshClusterLodProc meshProc;
  MeshletConeCullProc coneCullProc;

  MeshDescriptor meshDesc;
  meshDesc.indexCount = size_cast<int>(meshData->m_indices.size());
  meshDesc.indexData = reinterpret_cast<i8 *>(meshData->m_indices.data());
  meshDesc.positionOffset = 0;
  meshDesc.vertexCount = size_cast<int>(meshData->m_vertices.size());
  meshDesc.vertexData = reinterpret_cast<i8 *>(meshData->m_vertices.data());
  meshDesc.vertexStride = sizeof(ifloat3);
  meshDesc.normalData = reinterpret_cast<i8 *>(meshData->m_normals.data());
  meshDesc.normalStride = sizeof(ifloat3);

  auto chosenLod = MAX_LOD - 1;
  auto totalLods = 0;
  CombinedClusterLodBuffer meshletData;

  std::vector<FlattenedBVHNode> bvhNodes;
  std::vector<ClusterGroup> clusterGroupData;

  bool ableToLoadCachedVG = false;
  bool needToGenerateVG = false;
  bool needToStoreVG = false;

  auto serialCCLBufferName = "core.mesh.ccl." + meshData->identifier + ".cache";
  auto serialFBNName = "core.mesh.fbn." + meshData->identifier + ".cache";
  auto serialCGName = "core.mesh.cg." + meshData->identifier + ".cache";
  auto serialMiscName = "core.mesh.misc." + meshData->identifier + ".cache";

  auto serialCCLPath = cachePath + serialCCLBufferName;
  auto serialFBNPath = cachePath + serialFBNName;
  auto serialCGPath = cachePath + serialCGName;
  auto serialMiscPath = cachePath + serialMiscName;

  if (meshData->identifier.empty()) {
    needToGenerateVG = true;
  } else {
    if (cachePath.empty()) {
      needToGenerateVG = true;
    } else {
      bool cclBufferExists = false;
      bool fbnExists = false;
      bool cgExists = false;
      bool miscExists = false;

      // check files exist
      cclBufferExists = std::filesystem::exists(serialCCLPath);
      fbnExists = std::filesystem::exists(serialFBNPath);
      cgExists = std::filesystem::exists(serialCGPath);
      miscExists = std::filesystem::exists(serialMiscPath);

      if (cclBufferExists && fbnExists && cgExists && miscExists) {
        ableToLoadCachedVG = true;
      } else {
        needToGenerateVG = true;
        needToStoreVG = true;
      }
    }
  }

  if (ableToLoadCachedVG) {
    String cclBuffer;
    String fbnBuffer;
    String cgBuffer;
    String miscBuffer;

    cclBuffer = Ifrit::Common::Utility::readBinaryFile(serialCCLPath);
    fbnBuffer = Ifrit::Common::Utility::readBinaryFile(serialFBNPath);
    cgBuffer = Ifrit::Common::Utility::readBinaryFile(serialCGPath);
    miscBuffer = Ifrit::Common::Utility::readBinaryFile(serialMiscPath);

    Ifrit::Common::Serialization::deserializeBinary(cclBuffer, meshletData);
    Ifrit::Common::Serialization::deserializeBinary(fbnBuffer, bvhNodes);
    Ifrit::Common::Serialization::deserializeBinary(cgBuffer, clusterGroupData);
    CreateMeshLodHierMiscInfo miscInfo;
    Ifrit::Common::Serialization::deserializeBinary(miscBuffer, miscInfo);

    totalLods = miscInfo.totalLods;

    // iInfo("Loaded cached mesh VG for {}", meshData->identifier);
  }
  if (needToGenerateVG) {
    totalLods = meshProc.clusterLodHierachy(meshDesc, meshletData, clusterGroupData, bvhNodes, MAX_LOD);
    if (needToStoreVG) {
      String cclBuffer;
      Ifrit::Common::Serialization::serializeBinary(meshletData, cclBuffer);
      String fbnBuffer;
      Ifrit::Common::Serialization::serializeBinary(bvhNodes, fbnBuffer);
      String cgBuffer;
      Ifrit::Common::Serialization::serializeBinary(clusterGroupData, cgBuffer);
      CreateMeshLodHierMiscInfo miscInfo;
      miscInfo.totalLods = totalLods;
      String miscBuffer;
      Ifrit::Common::Serialization::serializeBinary(miscInfo, miscBuffer);

      Ifrit::Common::Utility::writeBinaryFile(serialCCLPath, cclBuffer);
      Ifrit::Common::Utility::writeBinaryFile(serialFBNPath, fbnBuffer);
      Ifrit::Common::Utility::writeBinaryFile(serialCGPath, cgBuffer);
      Ifrit::Common::Utility::writeBinaryFile(serialMiscPath, miscBuffer);
    }
  }

  coneCullProc.createNormalCones(meshDesc, meshletData.meshletsRaw, meshletData.meshletVertices,
                                 meshletData.meshletTriangles, meshData->m_normalsCone, meshData->m_normalsConeApex,
                                 meshData->m_boundSphere);

  auto meshlet_triangles = meshletData.meshletTriangles;
  auto meshlets = meshletData.meshletsRaw;
  auto meshlet_vertices = meshletData.meshletVertices;
  auto meshlet_count = meshlets.size();
  auto meshlet_cull = meshletData.meshletCull;
  auto meshlet_graphPart = meshletData.graphPartition;
  auto meshlet_inClusterGroup = meshletData.meshletsInClusterGroups;

  meshData->m_meshlets.resize(meshlets.size());
  for (auto i = 0; i < meshData->m_normalsCone.size(); i++) {
    meshData->m_meshlets[i].normalConeAxisCutoff = meshData->m_normalsCone[i];
    meshData->m_meshlets[i].normalConeApex = meshData->m_normalsConeApex[i];
    meshData->m_meshlets[i].boundSphere = meshData->m_boundSphere[i];
  }

  for (size_t i = 0; i < meshlets.size(); i++) {
    meshData->m_meshlets[i].vertexOffset = meshlets[i].x;
    meshData->m_meshlets[i].triangleOffset = meshlets[i].y;
    meshData->m_meshlets[i].vertexCount = meshlets[i].z;
    meshData->m_meshlets[i].triangleCount = meshlets[i].w;
    meshData->m_meshlets[i].selfErrorSphere = meshletData.selfErrorSphereW[i];
  }

  u32 totalTriangles = 0;
  for (size_t i = 0; i < meshlet_count; i++) {
    auto orgOffset = meshData->m_meshlets[i].triangleOffset;
    meshData->m_meshlets[i].triangleOffset = totalTriangles;
    totalTriangles += meshData->m_meshlets[i].triangleCount;
    for (size_t j = 0; j < meshData->m_meshlets[i].triangleCount; j++) {
      u32 packedTriangle = 0;
      packedTriangle |= meshlet_triangles[orgOffset + j * 3];
      packedTriangle |= meshlet_triangles[orgOffset + j * 3 + 1] << 8;
      packedTriangle |= meshlet_triangles[orgOffset + j * 3 + 2] << 16;
      meshData->m_meshletTriangles.push_back(packedTriangle);
    }
  }

  meshData->m_meshletVertices = std::move(meshlet_vertices);
  meshData->m_meshCullData = std::move(meshlet_cull);
  meshData->m_meshletInClusterGroup = std::move(meshlet_inClusterGroup);
  meshData->m_bvhNodes = std::move(bvhNodes);
  meshData->m_clusterGroups = std::move(clusterGroupData);
  meshData->m_numMeshletsEachLod = std::move(meshletData.numClustersEachLod);

  meshData->m_maxLod = totalLods;
}

IFRIT_APIDECL ifloat4 Mesh::getBoundingSphere(const std::vector<ifloat3> &vertices) {
  vfloat3 minv = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max()};
  vfloat3 maxv = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                  -std::numeric_limits<float>::max()};
  for (auto &v : vertices) {
    minv = min(minv, vfloat3{v.x, v.y, v.z});
    maxv = max(maxv, vfloat3{v.x, v.y, v.z});
  }
  auto center = (minv + maxv) * 0.5f;
  auto radius = length(maxv - center);
  return {center.x, center.y, center.z, radius};
}

} // namespace Ifrit::Core