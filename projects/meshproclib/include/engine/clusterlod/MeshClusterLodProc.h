#pragma once
#include <common/core/ApiConv.h>
#include <cstdint>
#include <meshoptimizer/src/meshoptimizer.h>
#include <vector>
#include <common/math/LinalgOps.h>

namespace Ifrit::Engine::MeshProcLib::ClusterLod {

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

struct ClusterLodGeneratorContext {
  int totalMeshlets;
  std::vector<meshopt_Meshlet> meshletsRaw;
  std::vector<uint32_t> meshletVertices;
  std::vector<uint8_t> meshletTriangles;
  std::vector<int32_t> graphPartition;
  std::vector<MeshletCullData> lodCullData;

  std::vector<uint32_t> parentStart;
  std::vector<uint32_t> parentSize;
};

struct CombinedClusterLodBuffer {
  std::vector<iint4> meshletsRaw; // 2x offsets + 2x size
  std::vector<uint32_t> meshletVertices;
  std::vector<uint8_t> meshletTriangles;
  std::vector<int32_t> graphPartition;
  std::vector<uint32_t> parentStart;
  std::vector<uint32_t> parentSize;
  std::vector<MeshletCullData> meshletCull;
};

class IFRIT_APIDECL MeshClusterLodProc {
public:
  int clusterLodHierachy(const MeshDescriptor &mesh,
                                 std::vector<ClusterLodGeneratorContext> &ctx,
                                 int maxLod);
  void combineLodData(const std::vector<ClusterLodGeneratorContext> &ctx,
                      CombinedClusterLodBuffer &out);
};

} // namespace Ifrit::Engine::MeshProcLib::ClusterLod