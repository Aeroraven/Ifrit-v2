
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
#include "MeshClusterBase.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include <cstdint>
#include <meshoptimizer/src/meshoptimizer.h>
#include <vector>

#ifndef IFRIT_MESHPROC_IMPORT
#define IFRIT_MESHPROC_API IFRIT_APIDECL
#else
#define IFRIT_MESHPROC_API IFRIT_APIDECL_IMPORT
#endif

namespace Ifrit::MeshProcLib::MeshProcess {

// This option disables DAG culling for cluster groups
#define IFRIT_MESHPROC_CLUSTERLOD_IGNORE_CLUSTERGROUP 0

struct ClusterLodGeneratorContext {
  i32 totalMeshlets;
  Vec<meshopt_Meshlet> meshletsRaw;
  Vec<u32> meshletVertices;
  Vec<u8> meshletTriangles;
  Vec<i32> graphPartition;
  Vec<MeshletCullData> lodCullData;
  Vec<Vector4f> selfErrorSphere;

  Vec<u32> parentStart;
  Vec<u32> parentSize;
  Vec<u32> childClusterId;

  // cluster groups
  Vec<ClusterGroup> clusterGroups;
  Vec<u32> meshletsInClusterGroups;
};

struct CombinedClusterLodBuffer {
  Vec<Vector4i> meshletsRaw; // 2x offsets + 2x size
  Vec<u32> meshletVertices;
  Vec<u8> meshletTriangles;
  Vec<i32> graphPartition;
  Vec<u32> parentStart;
  Vec<u32> parentSize;
  Vec<MeshletCullData> meshletCull;

  Vec<Vector4f> selfErrorSphereW;

  // cluster groups
  Vec<ClusterGroup> clusterGroups;
  Vec<u32> meshletsInClusterGroups;

  // Num clusters for each lod
  Vec<u32> numClustersEachLod;

  IFRIT_STRUCT_SERIALIZE(meshletsRaw, meshletVertices, meshletTriangles, graphPartition, parentStart, parentSize,
                         meshletCull, selfErrorSphereW, clusterGroups, meshletsInClusterGroups, numClustersEachLod);
};

class IFRIT_MESHPROC_API MeshClusterLodProc {
public:
  i32 clusterLodHierachy(const MeshDescriptor &mesh, CombinedClusterLodBuffer &meshletData,
                         Vec<ClusterGroup> &clusterGroupData, Vec<FlattenedBVHNode> &flattenedNodes, i32 maxLod);
};

} // namespace Ifrit::MeshProcLib::MeshProcess