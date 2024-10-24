#pragma once
#include <common/core/ApiConv.h>
#include <cstdint>
#include <meshoptimizer/src/meshoptimizer.h>
#include <vector>

namespace Ifrit::Engine::MeshProcLib::ClusterLod {

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
  std::vector<uint32_t> parentStart;
  std::vector<uint32_t> parentSize;
};

class IFRIT_APIDECL MeshClusterLodProc {
public:
  int clusterLodHierachy(const MeshDescriptor &mesh,
                                 std::vector<ClusterLodGeneratorContext> &ctx,
                                 int maxLod);
};

} // namespace Ifrit::Engine::MeshProcLib::ClusterLod