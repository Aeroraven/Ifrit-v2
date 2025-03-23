
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

#include "ifrit/meshproc/engine/mesh/MeshClusterLodProc.h"
#include "ifrit/common/base/IfritBase.h"

#if IFRIT_FEATURE_SIMD
#include <emmintrin.h>
#define IFRIT_USE_SIMD_128 1
#endif

#if IFRIT_FEATURE_SIMD_AVX256
#include <immintrin.h>
#define IFRIT_USE_SIMD_256 1
#endif

#include "ifrit/common/logging/Logging.h"

#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/common/math/VectorOps.h"
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/common/util/TypingUtil.h"
#include <meshoptimizer/src/meshoptimizer.h>
#include <metis.h>

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Ifrit::Math::SIMD;

namespace Ifrit::MeshProcLib::MeshProcess {
constexpr int CLUSTER_GROUP_SIZE = 4;
constexpr int TRIANGLES_PER_MESHLET = 128;
constexpr int VERTICES_PER_MESHLET = 128;
constexpr float MESH_SIMPLIFICATION_RATE = 0.5f;

// Meshlet definition

struct SimplifiedMeshlet {
  Vec<u32> index;
  float simplifyError;
};

template <typename T> T ceilDiv(T a, T b) { return (a + b - 1) / b; }

uint64_t packUnorderedPair(u32 a, u32 b) {
  if (a > b) {
    std::swap(a, b);
  }
  return (uint64_t(a) << 32) | b;
}
std::tuple<u32, u32> unpackUnorderedPair(uint64_t pair) { return std::make_tuple(pair >> 32, pair & 0xFFFFFFFFull); }

void freeUnusedMemoryInCotenxt(ClusterLodGeneratorContext &ctx) {
  auto maxMeshlets = ctx.totalMeshlets;
  auto maxVertexCount = ctx.meshletsRaw[maxMeshlets - 1].vertex_offset + ctx.meshletsRaw[maxMeshlets - 1].vertex_count;
  auto maxTriangleCount =
      ctx.meshletsRaw[maxMeshlets - 1].triangle_offset + ctx.meshletsRaw[maxMeshlets - 1].triangle_count * 3;
  ctx.meshletsRaw.resize(maxMeshlets);
  ctx.meshletVertices.resize(maxVertexCount);
  ctx.meshletTriangles.resize(maxTriangleCount);
}
void freeUnusedMemoryInCotenxt2(Vec<meshopt_Meshlet> &meshletsRaw, Vec<u32> &meshletVertices,
                                Vec<uint8_t> &meshletTriangles, int totalMeshlets) {
  auto maxMeshlets = totalMeshlets;
  auto maxVertexCount = meshletsRaw[maxMeshlets - 1].vertex_offset + meshletsRaw[maxMeshlets - 1].vertex_count;
  auto maxTriangleCount =
      meshletsRaw[maxMeshlets - 1].triangle_offset + meshletsRaw[maxMeshlets - 1].triangle_count * 3;
  meshletsRaw.resize(maxMeshlets);
  meshletVertices.resize(maxVertexCount);
  meshletTriangles.resize(maxTriangleCount);
}

void getBoundingBoxForCluster(const MeshDescriptor &mesh, const Vec<u32> index, vfloat3 &bMin, vfloat3 &bMax) {
  bMax = vfloat3(-std::numeric_limits<float>::max());
  bMin = vfloat3(std::numeric_limits<float>::max());

  for (int i = 0; i < index.size(); i++) {
    ifloat3 *pos = reinterpret_cast<ifloat3 *>(mesh.vertexData + mesh.positionOffset + mesh.vertexStride * index[i]);
    bMax = max(bMax, vfloat3(pos->x, pos->y, pos->z));
    bMin = min(bMin, vfloat3(pos->x, pos->y, pos->z));
  }
}

// First step, decompose the mesh into meshlets. [Mesh->Meshlet]
void initialMeshletGeneration(const MeshDescriptor &mesh, ClusterLodGeneratorContext &ctx) {
  auto maxMeshlets = meshopt_buildMeshletsBound(mesh.indexCount, VERTICES_PER_MESHLET, TRIANGLES_PER_MESHLET);
  ctx.meshletsRaw.resize(maxMeshlets);
  ctx.meshletVertices.resize(maxMeshlets * VERTICES_PER_MESHLET);
  ctx.meshletTriangles.resize(maxMeshlets * TRIANGLES_PER_MESHLET * 3);

  auto meshletCount =
      meshopt_buildMeshlets(ctx.meshletsRaw.data(), ctx.meshletVertices.data(), ctx.meshletTriangles.data(),
                            reinterpret_cast<u32 *>(mesh.indexData), mesh.indexCount,
                            reinterpret_cast<float *>(mesh.vertexData + mesh.positionOffset), mesh.vertexCount,
                            mesh.vertexStride, VERTICES_PER_MESHLET, TRIANGLES_PER_MESHLET, 0.0f);

  ctx.totalMeshlets = Ifrit::Common::Utility::size_cast<int>(meshletCount);

  // write lod errors, 0 for initial meshlet
  ctx.lodCullData.resize(meshletCount);
  ctx.childClusterId.resize(meshletCount);
  for (int i = 0; i < meshletCount; i++) {
    Vec<u32> actualIndices;
    for (u32 j = 0; j < ctx.meshletsRaw[i].triangle_count; j++) {
      auto base = ctx.meshletsRaw[i].triangle_offset + j * 3;
      actualIndices.push_back(ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 0]]);
      actualIndices.push_back(ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 1]]);
      actualIndices.push_back(ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 2]]);
    }
    vfloat3 bMin, bMax;
    getBoundingBoxForCluster(mesh, actualIndices, bMin, bMax);
    float bRadius = length(bMax - bMin) * 0.5f;
    vfloat3 bCenter = (bMax + bMin) * 0.5f;
    ctx.lodCullData[i].selfSphere = ifloat4(bCenter.x, bCenter.y, bCenter.z, bRadius);
    ctx.lodCullData[i].selfError = 0.0f;
    ctx.childClusterId[i] = UINT32_MAX;

    // follows nanite, lod0's error radius should be negative
    ctx.selfErrorSphere.push_back(ifloat4(bCenter.x, bCenter.y, bCenter.z, -100.0f));
  }

  freeUnusedMemoryInCotenxt(ctx);
  // iDebug("Initial meshlet generation done, total meshlets:{}",
  //        ctx.totalMeshlets);
}

void metisValidation(ClusterLodGeneratorContext &ctx) {
  // for a cluster group, find whether child meshlets are adjacent
  std::map<u32, Vec<u32>> clusterGroupToMeshletMap;
  for (int i = 0; i < ctx.totalMeshlets; i++) {
    clusterGroupToMeshletMap[ctx.graphPartition[i]].push_back(i);
  }
  u32 a = 0;
  for (auto &[key, cluster] : clusterGroupToMeshletMap) {
    std::unordered_map<uint64_t, Vec<u32>> edgeToMeshletMap{};
    Vec<std::pair<int, int>> edges;
    for (auto i : cluster) {
      for (auto j : cluster) {
        if (i >= j)
          continue;
        std::unordered_set<uint64_t> edgeSetA;
        std::unordered_set<uint64_t> edgeSetB;
        for (u32 k = 0; k < ctx.meshletsRaw[i].triangle_count; k++) {
          auto base = ctx.meshletsRaw[i].triangle_offset + k * 3;
          auto a = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 0]];
          auto b = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 1]];
          auto c = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 2]];
          auto pair1 = packUnorderedPair(a, b);
          auto pair2 = packUnorderedPair(b, c);
          auto pair3 = packUnorderedPair(c, a);
          edgeSetA.insert(pair1);
          edgeSetA.insert(pair2);
          edgeSetA.insert(pair3);
        }

        for (u32 k = 0; k < ctx.meshletsRaw[j].triangle_count; k++) {
          auto base = ctx.meshletsRaw[j].triangle_offset + k * 3;
          auto a = ctx.meshletVertices[ctx.meshletsRaw[j].vertex_offset + ctx.meshletTriangles[base + 0]];
          auto b = ctx.meshletVertices[ctx.meshletsRaw[j].vertex_offset + ctx.meshletTriangles[base + 1]];
          auto c = ctx.meshletVertices[ctx.meshletsRaw[j].vertex_offset + ctx.meshletTriangles[base + 2]];
          auto pair1 = packUnorderedPair(a, b);
          auto pair2 = packUnorderedPair(b, c);
          auto pair3 = packUnorderedPair(c, a);
          edgeSetB.insert(pair1);
          edgeSetB.insert(pair2);
          edgeSetB.insert(pair3);
        }
        // find whether A and B intersection are empty
        bool intersect = false;
        for (auto edge : edgeSetA) {
          if (edgeSetB.find(edge) != edgeSetB.end()) {
            intersect = true;
            break;
          }
        }
        if (intersect) {
          edges.push_back({i, j});
          edges.push_back({j, i});
        }
      }
    }
    // then find num of connected subgraphs
    int numSubGraphs = 0;
    Vec<u32> visited(cluster.size(), 0);
    for (int k = 0; k < cluster.size(); k++) {
      if (!visited[k]) {
        auto graphSize = 0;
        Vec<u32> q;
        q.push_back(k);
        visited[k] = true;
        while (!q.empty()) {
          auto v = q.back();
          q.pop_back();
          for (int i = 0; i < cluster.size(); i++) {
            if (visited[i] ||
                std::find(edges.begin(), edges.end(), std::make_pair(cluster[v], cluster[i])) == edges.end())
              continue;
            q.push_back(i);
            visited[i] = 1;
          }
          graphSize++;
        }
        numSubGraphs++;
      }
    }
    if (numSubGraphs != 1) {
      // iWarn("After METIS cut, cluster group #{}, have {} disconnected "
      //       "subgraphs.",
      //       key, numSubGraphs);
      a++;
    }
  }
  if (a)
    iWarn("Total METIS abnormalities found in graph cut: {}", a);
}

void connectivityCheck(const Vec<Vec<int>> &adj) {
  Vec<u32> visited(adj.size(), 0);
  Vec<u32> q;

  for (int k = 0; k < adj.size(); k++) {
    if (!visited[k]) {
      auto graphSize = 0;
      q.push_back(k);
      visited[k] = true;

      while (!q.empty()) {
        auto v = q.back();
        q.pop_back();
        for (int i = 0; i < adj[v].size(); i++) {
          if (visited[adj[v][i]])
            continue;
          q.push_back(adj[v][i]);
          visited[adj[v][i]] = 1;
        }
        graphSize++;
      }
      // iDebug("Subgraph have: {}/{} nodes", graphSize, adj.size());
    }
  }
}

// Second step, find the adjacency information for each meshlet. [Meshlet->DAG]
void meshletAdjacencyGeneration(ClusterLodGeneratorContext &ctx) {
  // Build the adjacency information for each meshlet.
  std::unordered_map<i64, Vec<u32>> edgeToMeshletMap{};
  for (int i = 0; i < ctx.totalMeshlets; i++) {
    std::unordered_set<uint64_t> edgeSet;
    for (u32 j = 0; j < ctx.meshletsRaw[i].triangle_count; j++) {
      auto base = ctx.meshletsRaw[i].triangle_offset + j * 3;
      auto a = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 0]];
      auto b = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 1]];
      auto c = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 2]];
      auto pair1 = packUnorderedPair(a, b);
      auto pair2 = packUnorderedPair(b, c);
      auto pair3 = packUnorderedPair(c, a);
      edgeSet.insert(pair1);
      edgeSet.insert(pair2);
      edgeSet.insert(pair3);
    }
    for (auto edge : edgeSet) {
      edgeToMeshletMap[edge].push_back(i);
    }
  }
  Vec<Vec<i32>> meshletAdjacency(ctx.totalMeshlets);
  for (const auto &[edge, meshlets] : edgeToMeshletMap) {
    for (int i = 0; i < meshlets.size(); i++) {
      for (int j = i + 1; j < meshlets.size(); j++) {
        meshletAdjacency[meshlets[i]].push_back(meshlets[j]);
        meshletAdjacency[meshlets[j]].push_back(meshlets[i]);
      }
    }
  }

  // Build METIS xadj and adjncy arrays
  Vec<i32> xadj;
  Vec<i32> adjncy;
  Vec<i32> weights;
  xadj.push_back(0);
  for (int i = 0; i < meshletAdjacency.size(); i++) {
    Vec<i32> localWeights;
    Vec<i32> localAdjs;
    HashMap<i32, i32> weightMap;
    for (int j = 0; j < meshletAdjacency[i].size(); j++) {
      if (weightMap.find(meshletAdjacency[i][j]) == weightMap.end()) {
        weightMap[meshletAdjacency[i][j]] = Ifrit::Common::Utility::size_cast<int>(localWeights.size());
        localWeights.push_back(1);
        localAdjs.push_back(meshletAdjacency[i][j]);
      } else {
        localWeights[weightMap[meshletAdjacency[i][j]]]++;
      }
    }
    // add to weights and adjncy
    for (int j = 0; j < localWeights.size(); j++) {
      weights.push_back(localWeights[j]);
      adjncy.push_back(localAdjs[j]);
    }
    xadj.push_back(Ifrit::Common::Utility::size_cast<int>(adjncy.size()));
  }
  connectivityCheck(meshletAdjacency);
  // Call METIS
  idx_t nvtxs = ctx.totalMeshlets;
  idx_t ncon = 1;
  idx_t *xadjPtr = xadj.data();
  idx_t *adjncyPtr = adjncy.data();
  idx_t *vwgt = nullptr;
  idx_t *vsize = nullptr;
  idx_t *adjwgt = nullptr;
  idx_t nparts = std::max(2, ctx.totalMeshlets / CLUSTER_GROUP_SIZE);
  real_t *tpwgts = nullptr;
  real_t *ubvec = nullptr;
  idx_t edgeCut;
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_CCORDER] = 1;
  options[METIS_OPTION_SEED] = 42;
  options[METIS_OPTION_CONTIG] = 0;
  options[METIS_OPTION_NUMBERING] = 0;

  ctx.graphPartition.resize(ctx.totalMeshlets);

  auto result = METIS_PartGraphKway(&nvtxs, &ncon, xadjPtr, adjncyPtr, vwgt, vsize, adjwgt, &nparts, tpwgts, ubvec,
                                    options, &edgeCut, ctx.graphPartition.data());
  if (result != METIS_OK) {
    iError("METIS partition failed, error code: {}", result);
    std::abort();
  }
  // metisValidation(ctx);
}

// Third step, for each cluster group, generate an aggregated meshlet.
// and then simplify the meshlet. [DAG->IndexBuffer (Simplify)]
void clusterGroupSimplification(const MeshDescriptor &mesh, ClusterLodGeneratorContext &ctx,
                                ClusterLodGeneratorContext &outCtx, float predefError) {
  // Note details here:
  // https://www.reddit.com/r/GraphicsProgramming/comments/12jv49o/nanitelike_lods_experiments/
  // The parentError is calculated in cluster-group level,
  // ({meshlets})->(clusterGroup)=>simplifyError->(meshlets)
  // ====
  // Update to above question:
  // The parentError is calculated in cluster-group level,
  // The selfError is calculated in cluster level, (not in cluster-group level)

  Map<u32, Vec<u32>> clusterGroupToMeshletMap;
  for (int i = 0; i < ctx.totalMeshlets; i++) {
    clusterGroupToMeshletMap[ctx.graphPartition[i]].push_back(i);
  }

  // get scale
  auto modelScale = meshopt_simplifyScale(reinterpret_cast<f32 *>(mesh.vertexData + mesh.positionOffset),
                                          mesh.vertexCount, mesh.vertexStride);

  for (auto &[key, meshletsR] : clusterGroupToMeshletMap) {
    Vec<u32> aggregatedIndexBuffer;
    for (auto i : meshletsR) {
      // Copy the index buffer
      auto base = ctx.meshletsRaw[i].triangle_offset;
      auto count = ctx.meshletsRaw[i].triangle_count;
      for (int j = 0; j < static_cast<i32>(count); j++) {
        auto a = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + j * 3 + 0]];
        auto b = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + j * 3 + 1]];
        auto c = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + j * 3 + 2]];
        aggregatedIndexBuffer.push_back(a);
        aggregatedIndexBuffer.push_back(b);
        aggregatedIndexBuffer.push_back(c);
      }
    }

    // Simplify the index buffer
    Vec<u32> simplifiedIndexBuffer(aggregatedIndexBuffer.size());
    if (simplifiedIndexBuffer.size() == 0) {
      iInfo("MeshletR size: {}", meshletsR.size());
      for (auto i : meshletsR) {
        // Copy the index buffer
        auto base = ctx.meshletsRaw[i].triangle_offset;
        auto count = ctx.meshletsRaw[i].triangle_count;
        iInfo("Meshlet {} has {} triangles", i, count);
      }
      iError("Empty meshlet found in cluster group {}", key);
    }
    f32 simplifyError;
    // option: lockborder
    auto option = meshopt_SimplifyLockBorder | meshopt_SimplifyErrorAbsolute;
    auto targetIndexCount = static_cast<u32>(aggregatedIndexBuffer.size() * MESH_SIMPLIFICATION_RATE);
    f32 targetError = predefError * modelScale; // 0.01f;
    size_t simplifiedSize = 0.0;

    if (mesh.normalData == nullptr) {
      simplifiedSize =
          meshopt_simplify(simplifiedIndexBuffer.data(), aggregatedIndexBuffer.data(), aggregatedIndexBuffer.size(),
                           reinterpret_cast<f32 *>(mesh.vertexData + mesh.positionOffset), mesh.vertexCount,
                           mesh.vertexStride, targetIndexCount, targetError, option, &simplifyError);
    } else {
      float normalWeight[3] = {0.5f, 0.5f, 0.5f};
      simplifiedSize = meshopt_simplifyWithAttributes(
          simplifiedIndexBuffer.data(), aggregatedIndexBuffer.data(), aggregatedIndexBuffer.size(),
          reinterpret_cast<f32 *>(mesh.vertexData + mesh.positionOffset), mesh.vertexCount, mesh.vertexStride,
          reinterpret_cast<f32 *>(mesh.normalData), 3 * sizeof(f32), normalWeight, 3, nullptr, targetIndexCount,
          targetError, option, &simplifyError);
    }

    if (simplifiedSize == 0) {
      iError("Simplified size is 0, targetIndexCount: {}, targetError: {}", targetIndexCount, targetError);
      iInfo("SimplifiedIndexBuffer size: {}", simplifiedIndexBuffer.size());
      iInfo("AggregatedIndexBuffer size: {}", aggregatedIndexBuffer.size());
      std::abort();
    }
    simplifiedIndexBuffer.resize(simplifiedSize);

    auto targetErrorModel = simplifyError;

    // Split into meshlets, using meshopt_buildMeshlet
    Vec<meshopt_Meshlet> meshlets;
    Vec<MeshletCullData> meshletsCull;
    Vec<u32> meshletVertices;
    Vec<u8> meshletTriangles;

    auto maxMeshlets =
        meshopt_buildMeshletsBound(simplifiedIndexBuffer.size(), VERTICES_PER_MESHLET, TRIANGLES_PER_MESHLET);
    meshlets.resize(maxMeshlets);
    meshletVertices.resize(maxMeshlets * VERTICES_PER_MESHLET);
    meshletTriangles.resize(maxMeshlets * TRIANGLES_PER_MESHLET * 3);
    auto meshletCount = meshopt_buildMeshlets(
        meshlets.data(), meshletVertices.data(), meshletTriangles.data(), simplifiedIndexBuffer.data(),
        simplifiedIndexBuffer.size(), reinterpret_cast<f32 *>(mesh.vertexData + mesh.positionOffset), mesh.vertexCount,
        mesh.vertexStride, VERTICES_PER_MESHLET, TRIANGLES_PER_MESHLET, 0.0f);

    freeUnusedMemoryInCotenxt2(meshlets, meshletVertices, meshletTriangles,
                               Ifrit::Common::Utility::size_cast<u32>(meshletCount));

    vfloat3 bMin, bMax;
    getBoundingBoxForCluster(mesh, aggregatedIndexBuffer, bMin, bMax);
    f32 bRadius = length(bMax - bMin) * 0.5f;
    vfloat3 bCenter = (bMax + bMin) * 0.5f;

    // for lod x, set cluster group data, uses last lod data
    // write child nodes with parent data
    auto childLod = ctx.lodCullData[meshletsR[0]].lod;
    auto accumulatedError = 0.0f;
    for (auto i : meshletsR) {
      // accumulatedError = std::max(accumulatedError, ctx.lodCullData[i].selfError);
      accumulatedError = std::max(accumulatedError, ctx.selfErrorSphere[i].w);
    }
    targetErrorModel += accumulatedError;

    for (auto i : meshletsR) {
      ctx.lodCullData[i].parentSphere.x = bCenter.x;
      ctx.lodCullData[i].parentSphere.y = bCenter.y;
      ctx.lodCullData[i].parentSphere.z = bCenter.z;
      ctx.lodCullData[i].parentSphere.w = bRadius;
      ctx.lodCullData[i].parentError = targetErrorModel;
    }

    // write all child meshlets, and finally register the cluster group
    // coherent bounding sphere and errors should be ensured in one cluster
    // group

#if IFRIT_MESHPROC_CLUSTERLOD_IGNORE_CLUSTERGROUP
    ClusterGroup curClusterGroup;
    curClusterGroup.parentBoundingSphere = ifloat4(bCenter.x, bCenter.y, bCenter.z, bRadius);
    curClusterGroup.parentBoundingSphere.w = targetErrorModel;
    curClusterGroup.childMeshletSize = 1;

    for (auto i : meshletsR) {
      auto clusterGroupChildInsStart = ctx.meshletsInClusterGroups.size();
      curClusterGroup.childMeshletStart = Ifrit::Common::Utility::size_cast<u32>(clusterGroupChildInsStart);
      ctx.meshletsInClusterGroups.push_back(i);
      ctx.clusterGroups.push_back(curClusterGroup);
    }
#else
    ClusterGroup curClusterGroup;
    curClusterGroup.parentBoundingSphere = ifloat4(bCenter.x, bCenter.y, bCenter.z, bRadius);
    curClusterGroup.parentBoundingSphere.w = targetErrorModel;
    curClusterGroup.childMeshletSize = Ifrit::Common::Utility::size_cast<u32>(meshletsR.size());
    curClusterGroup.childMeshletStart = Ifrit::Common::Utility::size_cast<u32>(ctx.meshletsInClusterGroups.size());
    for (auto i : meshletsR) {
      ctx.meshletsInClusterGroups.push_back(i);
    }
    ctx.clusterGroups.push_back(curClusterGroup);
#endif

    meshletsCull.resize(meshletCount);

    for (int i = 0; i < meshletCount; i++) {
      meshletsCull[i].selfSphere.x = bCenter.x;
      meshletsCull[i].selfSphere.y = bCenter.y;
      meshletsCull[i].selfSphere.z = bCenter.z;
      meshletsCull[i].selfSphere.w = bRadius;
      meshletsCull[i].selfError = targetErrorModel;
      meshletsCull[i].lod = childLod + 1;
    }

    // Copy the meshlets to the output context
    auto newMeshletVertexSize = 0;
    auto newMeshletTriangleSize = 0;
    newMeshletVertexSize += meshlets[meshletCount - 1].vertex_offset + meshlets[meshletCount - 1].vertex_count;
    newMeshletTriangleSize +=
        meshlets[meshletCount - 1].triangle_offset + meshlets[meshletCount - 1].triangle_count * 3;
    auto newMeshletVertexOffset = outCtx.meshletVertices.size();
    auto newMeshletTriangleOffset = outCtx.meshletTriangles.size();
    auto newMeshletCount = outCtx.totalMeshlets + meshletCount;

    ctx.parentStart.push_back(outCtx.totalMeshlets);
    ctx.parentSize.push_back(Ifrit::Common::Utility::size_cast<u32>(meshletCount));
    outCtx.meshletVertices.resize(newMeshletVertexOffset + newMeshletVertexSize);
    outCtx.meshletTriangles.resize(newMeshletTriangleOffset + newMeshletTriangleSize);
    outCtx.meshletsRaw.resize(newMeshletCount);
    outCtx.lodCullData.resize(newMeshletCount);
    outCtx.childClusterId.resize(newMeshletCount);
    outCtx.selfErrorSphere.resize(newMeshletCount);

    // just std::insert
    for (int i = 0; i < meshletCount; i++) {
      auto base = meshlets[i].triangle_offset;
      auto count = meshlets[i].triangle_count;
      outCtx.meshletsRaw[outCtx.totalMeshlets + i] = meshlets[i];
      outCtx.meshletsRaw[outCtx.totalMeshlets + i].vertex_offset +=
          Ifrit::Common::Utility::size_cast<u32>(newMeshletVertexOffset);
      outCtx.meshletsRaw[outCtx.totalMeshlets + i].triangle_offset +=
          Ifrit::Common::Utility::size_cast<u32>(newMeshletTriangleOffset);
      outCtx.lodCullData[outCtx.totalMeshlets + i] = meshletsCull[i];
      outCtx.childClusterId[outCtx.totalMeshlets + i] = key;
      outCtx.selfErrorSphere[outCtx.totalMeshlets + i] = ifloat4(bCenter.x, bCenter.y, bCenter.z, targetErrorModel);
    }
    outCtx.meshletVertices.insert(outCtx.meshletVertices.begin() + newMeshletVertexOffset, meshletVertices.begin(),
                                  meshletVertices.begin() + newMeshletVertexSize);

    outCtx.meshletTriangles.insert(outCtx.meshletTriangles.begin() + newMeshletTriangleOffset, meshletTriangles.begin(),
                                   meshletTriangles.begin() + newMeshletTriangleSize);
    outCtx.totalMeshlets = Ifrit::Common::Utility::size_cast<int>(newMeshletCount);
    // break;
  }
}

// Finally, put all things together.
int generateClusterLodHierachy(const MeshDescriptor &mesh, Vec<ClusterLodGeneratorContext> &ctx, int maxLod) {
  ctx.resize(maxLod);
  initialMeshletGeneration(mesh, ctx[0]);
  for (int i = 1; i < maxLod; i++) {
    meshletAdjacencyGeneration(ctx[i - 1]);
    // Error ramping follows:
    // https://jglrxavpok.github.io/2024/01/19/recreating-nanite-lod-generation.html
    auto error = std::lerp(0.01f, 0.5f, i / float(maxLod));
    clusterGroupSimplification(mesh, ctx[i - 1], ctx[i], i / float(maxLod));
    if (ctx[i].totalMeshlets <= 1 || i == maxLod - 1) {
      // push parent
      for (int j = 0; j < ctx[i - 1].totalMeshlets; j++) {
        ctx[i].parentStart.push_back(-1);
        ctx[i].parentSize.push_back(-1);
      }
      return i;
    }
  }
  return maxLod;
}

void combineBuffer(const Vec<ClusterLodGeneratorContext> &ctx, CombinedClusterLodBuffer &outCtx) {

  int prevlevelMeshletCount = 0;
  int prevlevelVertexCount = 0;
  int prevlevelTriangleCount = 0;
  int prevlevelClusterGroupCount = 0;
  int prevlevelMeshletInGroupCount = 0;
  for (int i = 0; i < ctx.size(); i++) {
    // Reserve spaces
    outCtx.meshletsRaw.resize(prevlevelMeshletCount + ctx[i].totalMeshlets);
    outCtx.selfErrorSphereW.resize(prevlevelMeshletCount + ctx[i].totalMeshlets);
    outCtx.meshletCull.resize(prevlevelMeshletCount + ctx[i].totalMeshlets);
    outCtx.clusterGroups.resize(prevlevelClusterGroupCount + ctx[i].clusterGroups.size());
    outCtx.meshletsInClusterGroups.resize(prevlevelMeshletInGroupCount + ctx[i].meshletsInClusterGroups.size());
    if (ctx[i].meshletsInClusterGroups.size() != ctx[i].totalMeshlets) {
      iError("Inconsistent data");
      std::abort();
    }
    for (int j = 0; j < ctx[i].clusterGroups.size(); j++) {
      outCtx.clusterGroups[prevlevelClusterGroupCount + j] = ctx[i].clusterGroups[j];
      outCtx.clusterGroups[prevlevelClusterGroupCount + j].childMeshletStart += prevlevelMeshletInGroupCount;
      outCtx.clusterGroups[prevlevelClusterGroupCount + j].lod = i;
    }
    if (prevlevelMeshletInGroupCount != prevlevelMeshletCount) {
      iError("Inconsistent meshlet data");
      std::abort();
    }

    for (int j = 0; j < ctx[i].totalMeshlets; j++) {
      outCtx.meshletsRaw[prevlevelMeshletCount + j] =
          iint4(ctx[i].meshletsRaw[j].vertex_offset + prevlevelVertexCount,
                ctx[i].meshletsRaw[j].triangle_offset + prevlevelTriangleCount, ctx[i].meshletsRaw[j].vertex_count,
                ctx[i].meshletsRaw[j].triangle_count);
      outCtx.selfErrorSphereW[prevlevelMeshletCount + j] = ctx[i].selfErrorSphere[j];
      outCtx.meshletCull[prevlevelMeshletCount + j] = ctx[i].lodCullData[j];
    }
    outCtx.graphPartition.insert(outCtx.graphPartition.end(), ctx[i].graphPartition.begin(),
                                 ctx[i].graphPartition.end());
    outCtx.meshletVertices.insert(outCtx.meshletVertices.end(), ctx[i].meshletVertices.begin(),
                                  ctx[i].meshletVertices.end());
    outCtx.meshletTriangles.insert(outCtx.meshletTriangles.end(), ctx[i].meshletTriangles.begin(),
                                   ctx[i].meshletTriangles.end());
    outCtx.meshletsInClusterGroups.insert(outCtx.meshletsInClusterGroups.end(), ctx[i].meshletsInClusterGroups.begin(),
                                          ctx[i].meshletsInClusterGroups.end());
    for (int j = 0; j < ctx[i].meshletsInClusterGroups.size(); j++) {
      outCtx.meshletsInClusterGroups[prevlevelMeshletInGroupCount + j] =
          ctx[i].meshletsInClusterGroups[j] + prevlevelMeshletCount;
    }

    // TODO: Compat size
    prevlevelMeshletCount += Ifrit::Common::Utility::size_cast<i32>(ctx[i].meshletsRaw.size());
    prevlevelVertexCount += Ifrit::Common::Utility::size_cast<i32>(ctx[i].meshletVertices.size());
    prevlevelTriangleCount += Ifrit::Common::Utility::size_cast<i32>(ctx[i].meshletTriangles.size());
    prevlevelClusterGroupCount += Ifrit::Common::Utility::size_cast<i32>(ctx[i].clusterGroups.size());
    prevlevelMeshletInGroupCount += Ifrit::Common::Utility::size_cast<i32>(ctx[i].meshletsInClusterGroups.size());
  }
  iInfo("Total cluster groups: {}", outCtx.clusterGroups.size());
}

// Runtime LoD selection starts, using BVH8(or BVH4) to cull cluster groups

struct BoundingBox {
  vfloat3 mi;
  vfloat3 ma;
};

struct BoundingBoxPair {
  BoundingBox all;
  BoundingBox posOnly;
};

struct ClusterGroupBVHNode {
  std::array<Uref<ClusterGroupBVHNode>, BVH_CHILDREN> child;
  Vec<const ClusterGroup *> childClusterGroups;
  u32 isLeaf = 0;
  u32 curChildren = 0;
  u32 subTreeSize = 0;
  f32 maxClusterError = 0.0f;
  BoundingBox bbox;
};

struct ClusterGroupBVHNodeBuildData {
  ClusterGroupBVHNode *bvhNode;
  u32 clusterStart = 0;
  u32 clusterEnd = 0;
};

struct ClusterGroupBVH {
  Uref<ClusterGroupBVHNode> root;
};

// Some bvh utility functions
BoundingBoxPair getBoundingBox(const Vec<ClusterGroup> &clusterGroups, const Vec<u32> &clusterGroupIndices, u32 start,
                               u32 end) {
  vfloat3 bMaxA = vfloat3(-std::numeric_limits<float>::max());
  vfloat3 bMinA = vfloat3(std::numeric_limits<float>::max());
  vfloat3 bMaxP = vfloat3(-std::numeric_limits<float>::max());
  vfloat3 bMinP = vfloat3(std::numeric_limits<float>::max());

  for (int k = start; k < static_cast<i32>(end); k++) {
    auto i = clusterGroupIndices[k];
    vfloat3 sphereCenter = vfloat3(clusterGroups[i].parentBoundingSphere.x, clusterGroups[i].parentBoundingSphere.y,
                                   clusterGroups[i].parentBoundingSphere.z);
    vfloat3 maxPos = sphereCenter + clusterGroups[i].parentBoundingSphere.w;
    vfloat3 minPos = sphereCenter - clusterGroups[i].parentBoundingSphere.w;
    bMaxA = max(maxPos, bMaxA);
    bMinA = min(minPos, bMinA);
    bMaxP = max(sphereCenter, bMaxP);
    bMinP = min(sphereCenter, bMinP);
  }
  BoundingBox bboxA{.mi = bMinA, .ma = bMaxA};
  BoundingBox bboxP{.mi = bMinP, .ma = bMaxP};
  return BoundingBoxPair{.all = bboxA, .posOnly = bboxP};
}

int findLongestAxis(const BoundingBox &bbox) {
  auto diff = bbox.ma - bbox.mi;
  int ret = 0;
  auto answ = diff.x;
  for (int i = 1; i < 3; i++) {
    if (elementAt(diff, i) > answ) {
      ret = i;
      answ = elementAt(diff, i);
    }
  }
  return ret;
}

u32 nodeClustersPatition(const Vec<ClusterGroup> &clusterGroups, Vec<u32> indices, u32 start, u32 end, float mid,
                         u32 axis) {
  int l = start, r = end - 1;
  auto getElementCenter = [&](u32 idx) {
    auto sphere = clusterGroups[idx].parentBoundingSphere;
    return vfloat3(sphere.x, sphere.y, sphere.z);
  };
  while (l < r) {
    while (l < r && elementAt(getElementCenter(indices[l]), axis) < mid)
      l++;
    while (l < r && elementAt(getElementCenter(indices[r]), axis) >= mid)
      r--;
    if (l < r) {
      std::swap(indices[l], indices[r]);
      l++;
      r--;
    }
  }
  auto pivot = elementAt(getElementCenter(indices[l]), axis) < mid ? l : l - 1;
  return pivot;
}

u32 nodeClustersPatitionAlternative(const Vec<ClusterGroup> &clusterGroups, Vec<u32> indices, u32 start, u32 end,
                                    u32 axis) {
  Vec<float> candidates;
  for (int i = start; i < static_cast<int>(end); i++) {
    auto sphere = clusterGroups[indices[i]].parentBoundingSphere;
    candidates.push_back(elementAt(vfloat3(sphere.x, sphere.y, sphere.z), axis));
  }
  // sort and find median
  std::sort(candidates.begin(), candidates.end());
  auto mid = candidates[candidates.size() / 2];
  return nodeClustersPatition(clusterGroups, indices, start, end, mid, axis);
}

// this builds a BVH2
void initialBVHConstruction(ClusterGroupBVH &bvh, const Vec<ClusterGroup> &clusterGroups) {
  Vec<ClusterGroupBVHNodeBuildData> q;
  // first, make bvh root
  bvh.root = std::make_unique<ClusterGroupBVHNode>();
  Vec<u32> clusterGroupIndices(clusterGroups.size());
  for (int i = 0; i < clusterGroups.size(); i++)
    clusterGroupIndices[i] = i;

  ClusterGroupBVHNodeBuildData node;
  node.bvhNode = bvh.root.get();
  node.clusterStart = 0;
  node.clusterEnd = Ifrit::Common::Utility::size_cast<u32>(clusterGroupIndices.size());
  q.push_back(node);

  // iteratively build nodes
  u32 totalNodes = 0;
  while (!q.empty()) {
    totalNodes++;
    auto curNode = q.back();
    q.pop_back();
    auto bbox = getBoundingBox(clusterGroups, clusterGroupIndices, curNode.clusterStart, curNode.clusterEnd);
    curNode.bvhNode->bbox = bbox.all;
    if (curNode.clusterEnd - curNode.clusterStart <= 1) {
      // already a leaf for bvh2
      curNode.bvhNode->isLeaf = true;
      for (int i = curNode.clusterStart; i < static_cast<int>(curNode.clusterEnd); i++) {
        auto idx = clusterGroupIndices[i];
        curNode.bvhNode->childClusterGroups.push_back(&clusterGroups[idx]);
      }
      continue;
    }

    auto longestAxis = findLongestAxis(bbox.all);
    auto midPartition = elementAt((bbox.all.ma + bbox.all.mi) * 0.5, longestAxis);
    auto leftNodeRightBoundIncl = nodeClustersPatition(clusterGroups, clusterGroupIndices, curNode.clusterStart,
                                                       curNode.clusterEnd, midPartition, longestAxis);
    if (leftNodeRightBoundIncl + 1 == curNode.clusterStart || leftNodeRightBoundIncl + 1 == curNode.clusterEnd) {
      // failed to partition
      curNode.bvhNode->isLeaf = true;
      for (int i = curNode.clusterStart; i < static_cast<int>(curNode.clusterEnd); i++) {
        auto idx = clusterGroupIndices[i];
        curNode.bvhNode->childClusterGroups.push_back(&clusterGroups[idx]);
      }
      continue;
    }

    curNode.bvhNode->child[0] = std::make_unique<ClusterGroupBVHNode>();
    curNode.bvhNode->child[1] = std::make_unique<ClusterGroupBVHNode>();

    ClusterGroupBVHNodeBuildData leftSubTree, rightSubTree;
    leftSubTree.clusterStart = curNode.clusterStart;
    leftSubTree.clusterEnd = leftNodeRightBoundIncl + 1;
    leftSubTree.bvhNode = curNode.bvhNode->child[0].get();
    rightSubTree.clusterStart = leftNodeRightBoundIncl + 1;
    rightSubTree.clusterEnd = curNode.clusterEnd;
    rightSubTree.bvhNode = curNode.bvhNode->child[1].get();

    q.push_back(leftSubTree);
    q.push_back(rightSubTree);
  }
  iDebug("Raw BVH Nodes:{}", totalNodes);
}

// utils for bvh collapse
consteval int xlog2(int n) { return (n <= 1) ? 0 : 1 + xlog2(n / 2); }
consteval u32 getBvhCollapseExtraDepth() { return xlog2(BVH_CHILDREN); }

// collapse bvh2 into bvh4 or bvh8
void bvhCollapse(ClusterGroupBVH &bvh) {
  Vec<ClusterGroupBVHNode *> q;
  q.push_back(bvh.root.get());
  constexpr auto extDepth = getBvhCollapseExtraDepth();

  struct IndirectChildrenSet {
    // after collapse, these nodes will be deleted
    Vec<Uref<ClusterGroupBVHNode>> nodesToCollapse;
    Vec<Uref<ClusterGroupBVHNode>> childNodes;
    IndirectChildrenSet() {}
    IndirectChildrenSet &operator=(const IndirectChildrenSet &p) = delete;
    IndirectChildrenSet(const IndirectChildrenSet &p) = delete;
  };

  std::function<void(ClusterGroupBVHNode *, IndirectChildrenSet *, int)> getChild =
      [&](ClusterGroupBVHNode *node, IndirectChildrenSet *indSet, int depth) {
        if (node->isLeaf) {
          return;
        }
        auto leftPtr = std::move(node->child[0]);
        auto rightPtr = std::move(node->child[1]);
        auto leftPtrRaw = leftPtr.get(), rightPtrRaw = rightPtr.get();
        if (leftPtr->isLeaf || depth == extDepth - 1) {
          indSet->childNodes.push_back(std::move(leftPtr));
        } else {
          indSet->nodesToCollapse.push_back(std::move(leftPtr));
          getChild(leftPtrRaw, indSet, depth + 1);
        }
        if (rightPtr->isLeaf || depth == extDepth - 1) {
          indSet->childNodes.push_back(std::move(rightPtr));
        } else {
          indSet->nodesToCollapse.push_back(std::move(rightPtr));
          getChild(rightPtrRaw, indSet, depth + 1);
        }
      };
  u32 totalNodes = 0;
  while (!q.empty()) {
    totalNodes++;
    auto curNode = q.back();
    q.pop_back();
    if (curNode->isLeaf) {
      curNode->curChildren = 0;
      continue;
    }
    // fetch curNodes' x order children, ownerships are transferred
    // into the indirect set
    IndirectChildrenSet indChild;
    getChild(curNode, &indChild, 0);
    for (int i = 0; i < indChild.childNodes.size(); i++) {
      curNode->child[i] = std::move(indChild.childNodes[i]);
    }
    for (int i = 0; i < indChild.childNodes.size(); i++) {
      q.push_back(curNode->child[i].get());
    }
    curNode->curChildren = Ifrit::Common::Utility::size_cast<u32>(indChild.childNodes.size());
  }
  iDebug("Total Nodes, after collapse:{}", totalNodes);

  // After collapse, dfs for subtree size
  struct DFSRet {
    u32 subTreeSize = 0;
    float maxClusterError = 0.0f;
  };
  std::function<DFSRet(ClusterGroupBVHNode *)> dfsSubTreeSize = [&](ClusterGroupBVHNode *node) -> DFSRet {
    if (node->isLeaf) {
      DFSRet ret;
      node->subTreeSize = 1;
      for (auto &clusterGroup : node->childClusterGroups) {
        ret.maxClusterError = std::max(ret.maxClusterError, clusterGroup->selfBoundingSphere.w);
      }
      node->maxClusterError = ret.maxClusterError;
      return ret;
    }
    DFSRet ret;
    ret.subTreeSize = 1;
    for (u32 i = 0; i < node->curChildren; i++) {
      auto childRet = dfsSubTreeSize(node->child[i].get());
      ret.subTreeSize += childRet.subTreeSize;
      ret.maxClusterError = std::max(ret.maxClusterError, node->child[i]->maxClusterError);

      // merge all leaf nodes' cluster groups
      if (node->child[i]->isLeaf) {
        for (auto &clusterGroup : node->child[i]->childClusterGroups) {
          node->childClusterGroups.push_back(clusterGroup);
        }
        ret.subTreeSize -= 1;
      }
    }

    // remove all leaf nodes (node->child), using erase if
    Vec<Uref<ClusterGroupBVHNode>> newChildren;
    for (u32 i = 0; i < node->curChildren; i++) {
      if (!node->child[i]->isLeaf) {
        newChildren.push_back(std::move(node->child[i]));
      }
    }
    for (u32 i = 0; i < node->curChildren; i++) {
      node->child[i] = nullptr;
    }
    node->curChildren = Ifrit::Common::Utility::size_cast<u32>(newChildren.size());
    for (u32 i = 0; i < node->curChildren; i++) {
      node->child[i] = std::move(newChildren[i]);
    }

    node->subTreeSize = ret.subTreeSize;
    node->maxClusterError = ret.maxClusterError;
    return ret;
  };
  dfsSubTreeSize(bvh.root.get());
}

// Flatten the bvh to make it compatible with GPU

void bvhFlatten(const ClusterGroupBVH &bvh, Vec<FlattenedBVHNode> &flattenedNodes,
                Vec<ClusterGroup> &rearrangedClusterGroups) {

  std::queue<ClusterGroupBVHNode *> q;
  q.push(bvh.root.get());

  u32 childId = 1;

  while (!q.empty()) {
    auto curNode = q.front();
    q.pop();

    FlattenedBVHNode newNode{};
    auto sphereCenter = (curNode->bbox.ma + curNode->bbox.mi) * 0.5f;
    auto sphereRadius = length(curNode->bbox.ma - curNode->bbox.mi) * 0.5f;
    newNode.boundSphere = ifloat4(sphereCenter.x, sphereCenter.y, sphereCenter.z, static_cast<float>(sphereRadius));
    newNode.numChildNodes = curNode->curChildren;
    newNode.clusterGroupStart = Ifrit::Common::Utility::size_cast<u32>(rearrangedClusterGroups.size());
    newNode.clusterGroupSize = Ifrit::Common::Utility::size_cast<u32>(curNode->childClusterGroups.size());
    newNode.subTreeSize = curNode->subTreeSize;
    newNode.maxClusterError = curNode->maxClusterError;
    for (int i = 0; i < static_cast<int>(curNode->curChildren); i++) {
      newNode.childNodes[i] = childId++;
      q.push(curNode->child[i].get());
    }
    for (int i = 0; i < curNode->childClusterGroups.size(); i++) {
      rearrangedClusterGroups.push_back(*curNode->childClusterGroups[i]);
    }
    flattenedNodes.push_back(newNode);
  }
}

// Class defs
IFRIT_APIDECL int MeshClusterLodProc::clusterLodHierachy(const MeshDescriptor &mesh,
                                                         CombinedClusterLodBuffer &meshletData,
                                                         Vec<ClusterGroup> &clusterGroupData,
                                                         Vec<FlattenedBVHNode> &flattenedNodes, int maxLod) {

  Vec<ClusterLodGeneratorContext> ctx;
  auto p = generateClusterLodHierachy(mesh, ctx, maxLod + 1);
  ctx.pop_back();
  ctx.resize(p);
  for (auto i = 0; i < ctx.size(); i++) {
    meshletData.numClustersEachLod.push_back(Ifrit::Common::Utility::size_cast<u32>(ctx[i].meshletsRaw.size()));
  }
  combineBuffer(ctx, meshletData);
  ClusterGroupBVH bvh;
  initialBVHConstruction(bvh, meshletData.clusterGroups);
  bvhCollapse(bvh);
  bvhFlatten(bvh, flattenedNodes, clusterGroupData);
  return p;
}

} // namespace Ifrit::MeshProcLib::MeshProcess