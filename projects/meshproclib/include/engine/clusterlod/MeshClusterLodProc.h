#pragma once
#include <common/core/ApiConv.h>
#include <common/math/LinalgOps.h>
#include <cstdint>
#include <meshoptimizer/src/meshoptimizer.h>
#include <vector>
 
namespace Ifrit::Engine::MeshProcLib::ClusterLod {
constexpr int BVH_CHILDREN = 8; // or 4
struct MeshDescriptor {
  char *vertexData;
  char *indexData;
  int vertexCount;
  int indexCount;
  int vertexStride;
  int positionOffset;
};

struct MeshletCullData {
  ifloat4 selfSphere;
  ifloat4 parentSphere;
  float selfError = INFINITY;
  float parentError = INFINITY;
  uint32_t lod = 0;
  uint32_t dummy = 0;
};

struct ClusterGroup {
  ifloat4 selfBoundingSphere;
  ifloat4 parentBoundingSphere; // No need for this, maybe
  float selfBoundError;
  float parentBoundError;
  uint32_t childMeshletStart;
  uint32_t childMeshletSize;
  uint32_t lod;
  uint32_t dummy1;
  uint32_t dummy2;
  uint32_t dummy3;
};

struct ClusterLodGeneratorContext {
  int totalMeshlets;
  std::vector<meshopt_Meshlet> meshletsRaw;
  std::vector<uint32_t> meshletVertices;
  std::vector<uint8_t> meshletTriangles;
  std::vector<int32_t> graphPartition;
  std::vector<MeshletCullData> lodCullData;

  std::vector<uint32_t> parentStart;
  std::vector<uint32_t> parentSize;
  std::vector<uint32_t> childClusterId;

  // cluster groups
  std::vector<ClusterGroup> clusterGroups;
  std::vector<uint32_t> meshletsInClusterGroups;
};

struct CombinedClusterLodBuffer {
  std::vector<iint4> meshletsRaw; // 2x offsets + 2x size
  std::vector<uint32_t> meshletVertices;
  std::vector<uint8_t> meshletTriangles;
  std::vector<int32_t> graphPartition;
  std::vector<uint32_t> parentStart;
  std::vector<uint32_t> parentSize;
  std::vector<MeshletCullData> meshletCull;

  // cluster groups
  std::vector<ClusterGroup> clusterGroups;
  std::vector<uint32_t> meshletsInClusterGroups;
};

struct FlattenedBVHNode {
  ifloat4 boundSphere;
  uint32_t numChildNodes;
  uint32_t clusterGroupStart;
  uint32_t clusterGroupSize;
  uint32_t subTreeSize;
  uint32_t childNodes[BVH_CHILDREN];
};

class IFRIT_APIDECL MeshClusterLodProc {
public:
  int clusterLodHierachy(const MeshDescriptor &mesh,
                         CombinedClusterLodBuffer &meshletData,
                         std::vector<ClusterGroup> &clusterGroupData,
                         std::vector<FlattenedBVHNode> &flattenedNodes,int maxLod);
};

} // namespace Ifrit::Engine::MeshProcLib::ClusterLod