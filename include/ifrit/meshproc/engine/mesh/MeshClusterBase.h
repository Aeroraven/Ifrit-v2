#pragma once
#include "ifrit/common/math/LinalgOps.h"

namespace Ifrit::MeshProcLib::MeshProcess {
constexpr int BVH_CHILDREN = 8; // or 4
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
struct FlattenedBVHNode {
  ifloat4 boundSphere;
  uint32_t numChildNodes;
  uint32_t clusterGroupStart;
  uint32_t clusterGroupSize;
  uint32_t subTreeSize;
  uint32_t childNodes[BVH_CHILDREN];
  float maxClusterError;
  uint32_t pad1;
  uint32_t pad2;
  uint32_t pad3;
};
struct MeshDescriptor {
  char *vertexData;
  char *indexData;
  char *normalData;
  int vertexCount;
  int indexCount;
  int vertexStride;
  int positionOffset;
  int normalStride;
};

} // namespace Ifrit::MeshProcLib::MeshProcess