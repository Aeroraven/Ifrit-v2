
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

struct ClusterLodGeneratorContext {
  int totalMeshlets;
  std::vector<meshopt_Meshlet> meshletsRaw;
  std::vector<u32> meshletVertices;
  std::vector<u8> meshletTriangles;
  std::vector<i32> graphPartition;
  std::vector<MeshletCullData> lodCullData;

  std::vector<u32> parentStart;
  std::vector<u32> parentSize;
  std::vector<u32> childClusterId;

  // cluster groups
  std::vector<ClusterGroup> clusterGroups;
  std::vector<u32> meshletsInClusterGroups;
};

struct CombinedClusterLodBuffer {
  std::vector<iint4> meshletsRaw; // 2x offsets + 2x size
  std::vector<u32> meshletVertices;
  std::vector<u8> meshletTriangles;
  std::vector<i32> graphPartition;
  std::vector<u32> parentStart;
  std::vector<u32> parentSize;
  std::vector<MeshletCullData> meshletCull;

  // cluster groups
  std::vector<ClusterGroup> clusterGroups;
  std::vector<u32> meshletsInClusterGroups;

  // Num clusters for each lod
  std::vector<u32> numClustersEachLod;

  IFRIT_STRUCT_SERIALIZE(meshletsRaw, meshletVertices, meshletTriangles, graphPartition, parentStart, parentSize,
                         meshletCull, clusterGroups, meshletsInClusterGroups, numClustersEachLod);
};

class IFRIT_MESHPROC_API MeshClusterLodProc {
public:
  int clusterLodHierachy(const MeshDescriptor &mesh, CombinedClusterLodBuffer &meshletData,
                         std::vector<ClusterGroup> &clusterGroupData, std::vector<FlattenedBVHNode> &flattenedNodes,
                         int maxLod);
};

} // namespace Ifrit::MeshProcLib::MeshProcess