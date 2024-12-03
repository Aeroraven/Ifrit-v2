
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