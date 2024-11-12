#include "ifrit/core/base/Mesh.h"
#define IFRIT_MESHPROC_IMPORT
#include "ifrit/meshproc/engine/clusterlod/MeshClusterLodProc.h"
#undef IFRIT_MESHPROC_IMPORT

namespace Ifrit::Core {
IFRIT_APIDECL void
Mesh::createMeshLodHierarchy(std::shared_ptr<MeshData> meshData) {
  using namespace Ifrit::MeshProcLib::ClusterLod;
  using namespace Ifrit::Common::Utility;
  const size_t max_vertices = 64;
  const size_t max_triangles = 124;
  const float cone_weight = 0.0f;
  constexpr int MAX_LOD = 4;

  MeshClusterLodProc meshProc;
  MeshDescriptor meshDesc;
  meshDesc.indexCount = size_cast<int>(meshData->m_indices.size());
  meshDesc.indexData = reinterpret_cast<char *>(meshData->m_indices.data());
  meshDesc.positionOffset = 0;
  meshDesc.vertexCount = size_cast<int>(meshData->m_vertices.size());
  meshDesc.vertexData = reinterpret_cast<char *>(meshData->m_vertices.data());
  meshDesc.vertexStride = sizeof(ifloat3);

  auto chosenLod = MAX_LOD - 1;
  CombinedClusterLodBuffer meshletData;

  std::vector<FlattenedBVHNode> bvhNodes;
  std::vector<ClusterGroup> clusterGroupData;

  meshProc.clusterLodHierachy(meshDesc, meshletData, clusterGroupData, bvhNodes,
                              MAX_LOD);

  auto meshlet_triangles = meshletData.meshletTriangles;
  auto meshlets = meshletData.meshletsRaw;
  auto meshlet_vertices = meshletData.meshletVertices;
  auto meshlet_count = meshlets.size();
  auto meshlet_cull = meshletData.meshletCull;
  auto meshlet_graphPart = meshletData.graphPartition;
  auto meshlet_inClusterGroup = meshletData.meshletsInClusterGroups;

  meshData->m_meshlets.resize(meshlets.size());
  for (size_t i = 0; i < meshlets.size(); i++) {
    meshData->m_meshlets[i] = meshlets[i];
  }
  for (size_t i = 0; i < meshlet_triangles.size(); i++) {
    meshData->m_meshletTriangles.push_back(meshlet_triangles[i]);
  }
  meshData->m_meshletVertices = std::move(meshlet_vertices);
  meshData->m_meshCullData = std::move(meshlet_cull);
  meshData->m_meshletInClusterGroup = std::move(meshlet_inClusterGroup);
  meshData->m_bvhNodes = std::move(bvhNodes);
  meshData->m_clusterGroups = std::move(clusterGroupData);

  meshData->m_maxLod = MAX_LOD;
}
} // namespace Ifrit::Core