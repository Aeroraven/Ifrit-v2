
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"

namespace Ifrit::MeshProcLib::MeshProcess {
constexpr int BVH_CHILDREN = 8; // or 4
struct MeshletCullData {
  ifloat4 selfSphere;
  ifloat4 parentSphere;
  float selfError = INFINITY;
  float parentError = INFINITY;
  u32 lod = 0;
  u32 dummy = 0;

  IFRIT_STRUCT_SERIALIZE(selfSphere, parentSphere, selfError, parentError, lod);
};
struct ClusterGroup {
  ifloat4 selfBoundingSphere;
  ifloat4 parentBoundingSphere; // No need for this, maybe
  u32 childMeshletStart;
  u32 childMeshletSize;
  u32 lod;
  u32 dummy1;

  IFRIT_STRUCT_SERIALIZE(selfBoundingSphere, parentBoundingSphere, childMeshletStart, childMeshletSize, lod);
};
struct FlattenedBVHNode {
  ifloat4 boundSphere;
  u32 numChildNodes;
  u32 clusterGroupStart;
  u32 clusterGroupSize;
  u32 subTreeSize;
  u32 childNodes[BVH_CHILDREN];
  float maxClusterError;
  u32 pad1;
  u32 pad2;
  u32 pad3;

  IFRIT_STRUCT_SERIALIZE(boundSphere, numChildNodes, clusterGroupStart, clusterGroupSize, subTreeSize, childNodes,
                         maxClusterError);
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