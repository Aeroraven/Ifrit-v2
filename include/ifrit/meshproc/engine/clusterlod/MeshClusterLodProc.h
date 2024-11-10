#pragma once
#include "MeshClusterBase.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/ApiConv.h"
#include <cstdint>
#include <meshoptimizer/src/meshoptimizer.h>
#include <vector>


#ifndef IFRIT_MESHPROC_IMPORT
#define IFRIT_MESHPROC_API IFRIT_APIDECL
#else
#define IFRIT_MESHPROC_API IFRIT_APIDECL_IMPORT
#endif

namespace Ifrit::MeshProcLib::ClusterLod {

struct MeshDescriptor {
  char *vertexData;
  char *indexData;
  int vertexCount;
  int indexCount;
  int vertexStride;
  int positionOffset;
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


class IFRIT_MESHPROC_API MeshClusterLodProc {
public:
  int clusterLodHierachy(const MeshDescriptor &mesh,
                         CombinedClusterLodBuffer &meshletData,
                         std::vector<ClusterGroup> &clusterGroupData,
                         std::vector<FlattenedBVHNode> &flattenedNodes,
                         int maxLod);
};

} // namespace Ifrit::MeshProcLib::ClusterLod