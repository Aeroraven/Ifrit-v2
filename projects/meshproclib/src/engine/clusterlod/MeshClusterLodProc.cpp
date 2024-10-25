#include <meshproclib/include/engine/clusterlod/MeshClusterLodProc.h>

#if IFRIT_FEATURE_SIMD
#include <emmintrin.h>
#define IFRIT_USE_SIMD_128 1
#endif

#if IFRIT_FEATURE_SIMD_AVX256
#include <immintrin.h>
#define IFRIT_USE_SIMD_256 1
#endif

#include <common/math/VectorDefs.h>
#include <common/math/VectorOps.h>
#include <common/math/simd/SimdVectors.h>
#include <meshoptimizer/src/meshoptimizer.h>
#include <metis.h>

#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Ifrit::Engine::MeshProcLib::ClusterLod {
constexpr int CLUSTER_GROUP_SIZE = 4;
constexpr int TRIANGLES_PER_MESHLET = 124;
constexpr int VERTICES_PER_MESHLET = 64;
constexpr float MESH_SIMPLIFICATION_RATE = 0.5f;

// Meshlet definition

struct SimplifiedMeshlet {
  std::vector<uint32_t> index;
  float simplifyError;
};

template <typename T> T ceilDiv(T a, T b) { return (a + b - 1) / b; }

uint64_t packUnorderedPair(uint32_t a, uint32_t b) {
  if (a > b) {
    std::swap(a, b);
  }
  return (uint64_t(a) << 32) | b;
}
std::tuple<uint32_t, uint32_t> unpackUnorderedPair(uint64_t pair) {
  return std::make_tuple(pair >> 32, pair & 0xFFFFFFFF);
}

void freeUnusedMemoryInCotenxt(ClusterLodGeneratorContext &ctx) {
  auto maxMeshlets = ctx.totalMeshlets;
  auto maxVertexCount = ctx.meshletsRaw[maxMeshlets - 1].vertex_offset +
                        ctx.meshletsRaw[maxMeshlets - 1].vertex_count;
  auto maxTriangleCount = ctx.meshletsRaw[maxMeshlets - 1].triangle_offset +
                          ctx.meshletsRaw[maxMeshlets - 1].triangle_count * 3;
  ctx.meshletsRaw.resize(maxMeshlets);
  ctx.meshletVertices.resize(maxVertexCount);
  ctx.meshletTriangles.resize(maxTriangleCount);
}
void freeUnusedMemoryInCotenxt2(std::vector<meshopt_Meshlet> &meshletsRaw,
                                std::vector<uint32_t> &meshletVertices,
                                std::vector<uint8_t> &meshletTriangles,
                                int totalMeshlets) {
  auto maxMeshlets = totalMeshlets;
  auto maxVertexCount = meshletsRaw[maxMeshlets - 1].vertex_offset +
                        meshletsRaw[maxMeshlets - 1].vertex_count;
  auto maxTriangleCount = meshletsRaw[maxMeshlets - 1].triangle_offset +
                          meshletsRaw[maxMeshlets - 1].triangle_count * 3;
  meshletsRaw.resize(maxMeshlets);
  meshletVertices.resize(maxVertexCount);
  meshletTriangles.resize(maxTriangleCount);
}

// First step, decompose the mesh into meshlets. [Mesh->Meshlet]
void initialMeshletGeneration(const MeshDescriptor &mesh,
                              ClusterLodGeneratorContext &ctx) {
  auto maxMeshlets = meshopt_buildMeshletsBound(
      mesh.indexCount, VERTICES_PER_MESHLET, TRIANGLES_PER_MESHLET);
  ctx.meshletsRaw.resize(maxMeshlets);
  ctx.meshletVertices.resize(maxMeshlets * VERTICES_PER_MESHLET);
  ctx.meshletTriangles.resize(maxMeshlets * TRIANGLES_PER_MESHLET * 3);

  auto meshletCount = meshopt_buildMeshlets(
      ctx.meshletsRaw.data(), ctx.meshletVertices.data(),
      ctx.meshletTriangles.data(), reinterpret_cast<uint32_t *>(mesh.indexData),
      mesh.indexCount,
      reinterpret_cast<float *>(mesh.vertexData + mesh.positionOffset),
      mesh.vertexCount, mesh.vertexStride, VERTICES_PER_MESHLET,
      TRIANGLES_PER_MESHLET, 0.0f);

  ctx.totalMeshlets = meshletCount;

  // write lod errors, 0 for initial meshlet
  ctx.lodCullData.resize(meshletCount);
  ctx.childClusterId.resize(meshletCount);
  for (int i = 0; i < meshletCount; i++) {
    ctx.lodCullData[i].selfError = 0.0f;
    ctx.childClusterId[i] = UINT32_MAX;
  }

  freeUnusedMemoryInCotenxt(ctx);
  printf("Initial meshlet generated\n");
}

// Second step, find the adjacency information for each meshlet. [Meshlet->DAG]
void meshletAdjacencyGeneration(ClusterLodGeneratorContext &ctx) {
  // Build the adjacency information for each meshlet.
  std::unordered_map<uint64_t, std::vector<uint32_t>> edgeToMeshletMap;
  for (int i = 0; i < ctx.totalMeshlets; i++) {
    std::unordered_set<uint64_t> edgeSet;
    for (int j = 0; j < ctx.meshletsRaw[i].triangle_count; j++) {
      auto base = ctx.meshletsRaw[i].triangle_offset + j * 3;
      auto off1 =
          ctx.meshletsRaw[i].vertex_offset + ctx.meshletTriangles[base + 0];
      auto a = ctx.meshletVertices[off1];
      auto b = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset +
                                   ctx.meshletTriangles[base + 1]];
      auto c = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset +
                                   ctx.meshletTriangles[base + 2]];
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
  std::vector<std::vector<int>> meshletAdjacency(ctx.totalMeshlets);
  for (const auto &[edge, meshlets] : edgeToMeshletMap) {
    for (int i = 0; i < meshlets.size(); i++) {
      for (int j = i + 1; j < meshlets.size(); j++) {
        meshletAdjacency[meshlets[i]].push_back(meshlets[j]);
        meshletAdjacency[meshlets[j]].push_back(meshlets[i]);
      }
    }
  }

  // Build METIS xadj and adjncy arrays
  std::vector<int> xadj;
  std::vector<int> adjncy;
  xadj.push_back(0);
  for (int i = 0; i < meshletAdjacency.size(); i++) {
    for (int j = 0; j < meshletAdjacency[i].size(); j++) {
      adjncy.push_back(meshletAdjacency[i][j]);
    }
    xadj.push_back(adjncy.size());
  }

  // Call METIS
  idx_t nvtxs = ctx.totalMeshlets;
  idx_t ncon = 1;
  idx_t *xadjPtr = xadj.data();
  idx_t *adjncyPtr = adjncy.data();
  idx_t *vwgt = nullptr;
  idx_t *vsize = nullptr;
  idx_t *adjwgt = nullptr;
  idx_t nparts = ctx.totalMeshlets / CLUSTER_GROUP_SIZE;
  real_t *tpwgts = nullptr;
  real_t *ubvec = nullptr;
  idx_t edgeCut;
  idx_t options[METIS_NOPTIONS];
  METIS_SetDefaultOptions(options);
  options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_CCORDER] = 1;

  ctx.graphPartition.resize(ctx.totalMeshlets);
  METIS_PartGraphKway(&nvtxs, &ncon, xadjPtr, adjncyPtr, vwgt, vsize, adjwgt,
                      &nparts, tpwgts, ubvec, options, &edgeCut,
                      ctx.graphPartition.data());
}

// Third step, for each cluster group, generate an aggregated meshlet.
// and then simplify the meshlet. [DAG->IndexBuffer (Simplify)]
void clusterGroupSimplification(const MeshDescriptor &mesh,
                                ClusterLodGeneratorContext &ctx,
                                ClusterLodGeneratorContext &outCtx) {
  int clusterGroupCount = ctx.totalMeshlets / CLUSTER_GROUP_SIZE;
  std::map<uint32_t, std::vector<uint32_t>> clusterGroupToMeshletMap;
  for (int i = 0; i < ctx.totalMeshlets; i++) {
    clusterGroupToMeshletMap[ctx.graphPartition[i]].push_back(i);
  }

  // get scale
  auto modelScale = meshopt_simplifyScale(
      reinterpret_cast<float *>(mesh.vertexData + mesh.positionOffset),
      mesh.vertexCount, mesh.vertexStride);

  for (auto &[key, meshletsR] : clusterGroupToMeshletMap) {
    std::vector<uint32_t> aggregatedIndexBuffer;
    for (auto i : meshletsR) {
      // Copy the index buffer
      auto base = ctx.meshletsRaw[i].triangle_offset;
      auto count = ctx.meshletsRaw[i].triangle_count;
      for (int j = 0; j < count; j++) {
        auto a = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset +
                                     ctx.meshletTriangles[base + j * 3 + 0]];
        auto b = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset +
                                     ctx.meshletTriangles[base + j * 3 + 1]];
        auto c = ctx.meshletVertices[ctx.meshletsRaw[i].vertex_offset +
                                     ctx.meshletTriangles[base + j * 3 + 2]];
        aggregatedIndexBuffer.push_back(a);
        aggregatedIndexBuffer.push_back(b);
        aggregatedIndexBuffer.push_back(c);
      }
    }
    // Simplify the index buffer
    std::vector<uint32_t> simplifiedIndexBuffer(aggregatedIndexBuffer.size());
    float simplifyError;
    // option: lockborder
    auto option = meshopt_SimplifyLockBorder;
    auto targetIndexCount = static_cast<uint32_t>(aggregatedIndexBuffer.size() *
                                                  MESH_SIMPLIFICATION_RATE);
    float targetError = 0.01;
    auto simplifiedSize = meshopt_simplify(
        simplifiedIndexBuffer.data(), aggregatedIndexBuffer.data(),
        aggregatedIndexBuffer.size(),
        reinterpret_cast<float *>(mesh.vertexData + mesh.positionOffset),
        mesh.vertexCount, mesh.vertexStride, targetIndexCount, targetError,
        option, &simplifyError);

    simplifiedIndexBuffer.resize(simplifiedSize);

    auto targetErrorModel = targetError * modelScale;

    // Split into meshlets, using meshopt_buildMeshlet
    std::vector<meshopt_Meshlet> meshlets;
    std::vector<MeshletCullData> meshletsCull;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint8_t> meshletTriangles;

    auto maxMeshlets =
        meshopt_buildMeshletsBound(simplifiedIndexBuffer.size(),
                                   VERTICES_PER_MESHLET, TRIANGLES_PER_MESHLET);
    meshlets.resize(maxMeshlets);
    meshletVertices.resize(maxMeshlets * VERTICES_PER_MESHLET);
    meshletTriangles.resize(maxMeshlets * TRIANGLES_PER_MESHLET * 3);
    auto meshletCount = meshopt_buildMeshlets(
        meshlets.data(), meshletVertices.data(), meshletTriangles.data(),
        simplifiedIndexBuffer.data(), simplifiedIndexBuffer.size(),
        reinterpret_cast<float *>(mesh.vertexData + mesh.positionOffset),
        mesh.vertexCount, mesh.vertexStride, VERTICES_PER_MESHLET,
        TRIANGLES_PER_MESHLET, 0.0f);

    freeUnusedMemoryInCotenxt2(meshlets, meshletVertices, meshletTriangles,
                               meshletCount);

    auto boundSphere = meshopt_computeClusterBounds(
        simplifiedIndexBuffer.data(), simplifiedIndexBuffer.size(),
        reinterpret_cast<float *>(mesh.vertexData + mesh.positionOffset),
        mesh.vertexCount, mesh.vertexStride);

    // write child nodes with parent data
    auto childLod = ctx.lodCullData[meshletsR[0]].lod;

    auto accumulatedError = 0.0f;
    for (auto i : meshletsR) {
      accumulatedError =
          std::max(accumulatedError, ctx.lodCullData[i].selfError);
    }
    targetErrorModel += accumulatedError;

    for (auto i : meshletsR) {
      ctx.lodCullData[i].parentSphere.x = boundSphere.center[0];
      ctx.lodCullData[i].parentSphere.y = boundSphere.center[1];
      ctx.lodCullData[i].parentSphere.z = boundSphere.center[2];
      ctx.lodCullData[i].parentSphere.w = boundSphere.radius;
      ctx.lodCullData[i].parentError = targetErrorModel;
    }

    meshletsCull.resize(meshletCount);

    for (int i = 0; i < meshletCount; i++) {
      meshletsCull[i].selfSphere.x = boundSphere.center[0];
      meshletsCull[i].selfSphere.y = boundSphere.center[1];
      meshletsCull[i].selfSphere.z = boundSphere.center[2];
      meshletsCull[i].selfSphere.w = boundSphere.radius;
      meshletsCull[i].selfError = targetErrorModel;
      meshletsCull[i].lod = childLod + 1;
    }

    // Copy the meshlets to the output context
    auto newMeshletVertexSize = 0;
    auto newMeshletTriangleSize = 0;
    newMeshletVertexSize += meshlets[meshletCount - 1].vertex_offset +
                            meshlets[meshletCount - 1].vertex_count;
    newMeshletTriangleSize += meshlets[meshletCount - 1].triangle_offset +
                              meshlets[meshletCount - 1].triangle_count * 3;
    auto newMeshletVertexOffset = outCtx.meshletVertices.size();
    auto newMeshletTriangleOffset = outCtx.meshletTriangles.size();
    auto newMeshletCount = outCtx.totalMeshlets + meshletCount;
    ctx.parentStart.push_back(outCtx.totalMeshlets);
    ctx.parentSize.push_back(meshletCount);
    outCtx.meshletVertices.resize(newMeshletVertexOffset +
                                  newMeshletVertexSize);
    outCtx.meshletTriangles.resize(newMeshletTriangleOffset +
                                   newMeshletTriangleSize);
    outCtx.meshletsRaw.resize(newMeshletCount);
    outCtx.lodCullData.resize(newMeshletCount);
    outCtx.childClusterId.resize(newMeshletCount);

    // just std::insert
    for (int i = 0; i < meshletCount; i++) {
      auto base = meshlets[i].triangle_offset;
      auto count = meshlets[i].triangle_count;
      outCtx.meshletsRaw[outCtx.totalMeshlets + i] = meshlets[i];
      outCtx.meshletsRaw[outCtx.totalMeshlets + i].vertex_offset +=
          newMeshletVertexOffset;
      outCtx.meshletsRaw[outCtx.totalMeshlets + i].triangle_offset +=
          newMeshletTriangleOffset;
      outCtx.lodCullData[outCtx.totalMeshlets + i] = meshletsCull[i];
      outCtx.childClusterId[outCtx.totalMeshlets + i] = key;
    }
    outCtx.meshletVertices.insert(
        outCtx.meshletVertices.begin() + newMeshletVertexOffset,
        meshletVertices.begin(),
        meshletVertices.begin() + newMeshletVertexSize);

    printf("Simplified cluster group: %d, localError=%f, center=%f,%f,%f\n",
           key, targetErrorModel, boundSphere.center[0], boundSphere.center[1],
           boundSphere.center[2]);
    outCtx.meshletTriangles.insert(
        outCtx.meshletTriangles.begin() + newMeshletTriangleOffset,
        meshletTriangles.begin(),
        meshletTriangles.begin() + newMeshletTriangleSize);
    outCtx.totalMeshlets = newMeshletCount;
    // break;
  }
}

// Finally, put all things together.
int generateClusterLodHierachy(const MeshDescriptor &mesh,
                               std::vector<ClusterLodGeneratorContext> &ctx,
                               int maxLod) {
  ctx.resize(maxLod);
  initialMeshletGeneration(mesh, ctx[0]);
  for (int i = 1; i < maxLod; i++) {
    printf("Generating lod:%d\n", i);
    meshletAdjacencyGeneration(ctx[i - 1]);
    clusterGroupSimplification(mesh, ctx[i - 1], ctx[i]);
    if (ctx[i].totalMeshlets <= 1 || i == maxLod - 1) {
      // push parent
      for (int j = 0; j < ctx[i - 1].totalMeshlets; j++) {
        ctx[i].parentStart.push_back(-1);
        ctx[i].parentSize.push_back(-1);
      }
      return i + 1;
    }
  }
  return maxLod;
}

// Dynamic lod selection requires a tree traversal at the cluster level
int clusterLevelDataBuild(const std::vector<ClusterLodGeneratorContext> &ctx,
                          MeshletClusterInfo &out) {
  // TODO: Place meshlets in same cluster group in a contiguous memory
  printf("Build cluster-level data\n");
  std::vector<uint32_t> lodClusterStarts;
  lodClusterStarts.resize(ctx.size() + 1);
  auto maxlod = ctx.size() - 1;

  // precalculate num of meshlets at each lod
  std::vector<uint32_t> lodMeshletCount;
  std::vector<uint32_t> lodClusterCountPrefixSum;
  uint32_t psum = 0;
  for (int i = 0; i < ctx.size(); i++) {
    lodMeshletCount.push_back(ctx[i].meshletsRaw.size());
    lodClusterCountPrefixSum.push_back(psum);
    psum += ctx[i].meshletsRaw.size();
  }

  // root lod
  MeshletClusterInfoBuffer rootData;
  rootData.subMeshletStart = 0;
  rootData.subMeshletCount = ctx[maxlod].totalMeshlets;
  rootData.childClusterStart = 0;
  rootData.childClusterCount = 0;
  rootData.boundingSphere = ifloat4(0.0f, 0.0f, 0.0f, INFINITY);
  out.clusterInfo.push_back(rootData);
  lodClusterStarts[ctx.size()] = 0;

  // get all clusters and their child meshlets
  auto addCluster = [&](int lod, int start, int count) {
    MeshletClusterInfoBuffer clusterData;
    clusterData.subMeshletStart = start + lodClusterCountPrefixSum[lod];
    clusterData.subMeshletCount = count;
    clusterData.childClusterStart = 0;
    clusterData.childClusterCount = 0;
    auto boundSphere = ctx[lod].lodCullData[start].selfSphere;
    clusterData.boundingSphere = boundSphere;
    out.clusterInfo.push_back(clusterData);
  };
  for (int lod = ctx.size() - 1; lod >= 0; lod--) {
    lodClusterStarts[lod] = out.clusterInfo.size();
    auto lastClusterId = UINT32_MAX;
    auto lastClusterStart = 0;
    for (int i = 0; i < ctx[lod].meshletsRaw.size(); i++) {
      auto clusterId = ctx[lod].childClusterId[i];
      if (clusterId != lastClusterId) {
        if (lastClusterId != UINT32_MAX) {
          addCluster(lod, lastClusterStart, i - lastClusterStart); 
        }
        lastClusterId = clusterId;
        lastClusterStart = i;
      }
    }
    if (lastClusterStart < ctx[lod].meshletsRaw.size()) {
      addCluster(lod, lastClusterStart,
                 ctx[lod].meshletsRaw.size() - lastClusterStart);
    }
  }
  // update root's child cluster count to max lod's
  auto dMaxLodStart = lodClusterStarts[maxlod];
  auto dMaxLodEnd =
      (maxlod == 0) ? out.clusterInfo.size() : lodClusterStarts[maxlod - 1];
  out.clusterInfo[0].childClusterCount = dMaxLodEnd - dMaxLodStart;
  out.clusterInfo[0].childClusterStart = dMaxLodStart;

  // build child cluster info
  std::vector<std::vector<uint32_t>>
      childClusterInfo; // at certain lod, cluster i owns child clusters j
  for (int lod = 0; lod < ctx.size() - 1; lod++) {
    auto dStart = lodClusterStarts[lod];
    auto dEnd = lod == 0 ? out.clusterInfo.size() : lodClusterStarts[lod - 1];

    // find how many lods are in the parent lod
    auto dParentStart = lodClusterStarts[lod + 1];
    auto dParentEnd = lodClusterStarts[lod];
    auto numParentClusters = dParentEnd - dParentStart;

    childClusterInfo.clear();
    childClusterInfo.resize(numParentClusters);
    for (int i = 0; i < ctx[lod].meshletsRaw.size(); i++) {
      auto clusterId = ctx[lod].childClusterId[i];
      auto parentClusterId = ctx[lod].graphPartition[i];
      auto &r = childClusterInfo[parentClusterId];
      r.push_back(clusterId + dStart);
    }
    for (int i = dParentStart; i < dParentEnd; i++) {
      auto &cluster = out.clusterInfo[i];
      auto parentClusterId = i - dParentStart;
      auto &childClusters = childClusterInfo[parentClusterId];
      cluster.childClusterStart = out.childClusters.size();
      cluster.childClusterCount = childClusters.size();
      out.childClusters.insert(out.childClusters.end(), childClusters.begin(),
                               childClusters.end());
    }
  }
  return 0;
}

void combineBuffer(const std::vector<ClusterLodGeneratorContext> &ctx,
                   CombinedClusterLodBuffer &outCtx) {

  int prevlevelMeshletCount = 0;
  int prevlevelVertexCount = 0;
  int prevlevelTriangleCount = 0;
  for (int i = 0; i < ctx.size(); i++) {
    // Reserve spaces
    outCtx.meshletsRaw.resize(prevlevelMeshletCount + ctx[i].totalMeshlets);
    outCtx.meshletCull.resize(prevlevelMeshletCount + ctx[i].totalMeshlets);

    for (int j = 0; j < ctx[i].totalMeshlets; j++) {
      outCtx.meshletsRaw[prevlevelMeshletCount + j] =
          iint4(ctx[i].meshletsRaw[j].vertex_offset + prevlevelVertexCount,
                ctx[i].meshletsRaw[j].triangle_offset + prevlevelTriangleCount,
                ctx[i].meshletsRaw[j].vertex_count,
                ctx[i].meshletsRaw[j].triangle_count);
      outCtx.meshletCull[prevlevelMeshletCount + j] = ctx[i].lodCullData[j];
    }
    outCtx.meshletVertices.insert(outCtx.meshletVertices.end(),
                                  ctx[i].meshletVertices.begin(),
                                  ctx[i].meshletVertices.end());
    outCtx.meshletTriangles.insert(outCtx.meshletTriangles.end(),
                                   ctx[i].meshletTriangles.begin(),
                                   ctx[i].meshletTriangles.end());
    // TODO: Compat size
    prevlevelMeshletCount += ctx[i].meshletsRaw.size();
    prevlevelVertexCount += ctx[i].meshletVertices.size();
    prevlevelTriangleCount += ctx[i].meshletTriangles.size();
    printf("Meshlet count at %d:%lld\n", i, ctx[i].meshletsRaw.size());
  }
}

IFRIT_APIDECL int MeshClusterLodProc::clusterLodHierachy(
    const MeshDescriptor &mesh, std::vector<ClusterLodGeneratorContext> &ctx,
    int maxLod) {
  return generateClusterLodHierachy(mesh, ctx, maxLod);
}

IFRIT_APIDECL void MeshClusterLodProc::combineLodData(
    const std::vector<ClusterLodGeneratorContext> &ctx,
    CombinedClusterLodBuffer &out) {
  return combineBuffer(ctx, out);
}
IFRIT_APIDECL void MeshClusterLodProc::clusterLodHierachyAll(
    const MeshDescriptor &mesh, CombinedClusterLodBuffer &meshletData,
    MeshletClusterInfo &clusterData, int maxlod) {
  std::vector<ClusterLodGeneratorContext> ctx;
  generateClusterLodHierachy(mesh, ctx, maxlod);
  combineBuffer(ctx, meshletData);
  clusterLevelDataBuild(ctx, clusterData);
}
} // namespace Ifrit::Engine::MeshProcLib::ClusterLod