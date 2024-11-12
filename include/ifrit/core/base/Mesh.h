#pragma once
#include "AssetReference.h"
#include "Component.h"
#include "Material.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/meshproc/engine/clusterlod/MeshClusterBase.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {
struct MeshData {
  struct GPUCPCounter {
    uint32_t consume;
    uint32_t produce;
    uint32_t remain;
    uint32_t totalBvhNodes;
    uint32_t totalNumClusters;
    uint32_t totalLods;
    uint32_t pad1;
    uint32_t pad2;
  };

  std::vector<ifloat3> m_vertices;
  std::vector<ifloat4> m_verticesAligned;
  std::vector<ifloat3> m_normals;
  std::vector<ifloat2> m_uvs;
  std::vector<ifloat3> m_tangents;
  std::vector<uint32_t> m_indices;

  // Cluster data
  std::vector<iint4> m_meshlets;
  std::vector<uint32_t> m_meshletTriangles;
  std::vector<uint32_t> m_meshletVertices;
  std::vector<uint32_t> m_meshletInClusterGroup;
  std::vector<Ifrit::MeshProcLib::ClusterLod::MeshletCullData> m_meshCullData;
  std::vector<Ifrit::MeshProcLib::ClusterLod::FlattenedBVHNode>
      m_bvhNodes; // seems not suitable to be here
  std::vector<Ifrit::MeshProcLib::ClusterLod::ClusterGroup> m_clusterGroups;

  GPUCPCounter m_cpCounter;
  uint32_t m_maxLod;

  IFRIT_STRUCT_SERIALIZE(m_vertices, m_normals, m_uvs, m_tangents, m_indices);
};

class IFRIT_APIDECL Mesh : public AssetReferenceContainer,
                           public IAssetCompatible {
  using GPUBuffer = Ifrit::GraphicsBackend::Rhi::RhiBuffer;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;

public:
  struct GPUObjectBuffer {
    uint32_t vertexBufferId;
    uint32_t meshletBufferId;
    uint32_t meshletVertexBufferId;
    uint32_t meshletIndexBufferId;
    uint32_t meshletCullBufferId;
    uint32_t bvhNodeBufferId;
    uint32_t clusterGroupBufferId;
    uint32_t meshletInClusterBufferId;
    uint32_t cpQueueBufferId;
    uint32_t cpCounterBufferId;
    uint32_t filteredMeshletsId;
    uint32_t pad;
  };

  struct GPUCPCounter {
    uint32_t consume;
    uint32_t produce;
    uint32_t remain;
    uint32_t totalBvhNodes;
    uint32_t totalNumClusters;
    uint32_t totalLods;
    uint32_t pad1;
    uint32_t pad2;
  };

  struct GPUResource {
    GPUBuffer *vertexBuffer = nullptr; // should be aligned
    GPUBuffer *meshletBuffer = nullptr;
    GPUBuffer *meshletVertexBuffer = nullptr;
    GPUBuffer *meshletIndexBuffer = nullptr;
    GPUBuffer *meshletCullBuffer = nullptr;
    GPUBuffer *bvhNodeBuffer = nullptr;
    GPUBuffer *clusterGroupBuffer = nullptr;
    GPUBuffer *meshletInClusterBuffer = nullptr;
    GPUBuffer *cpQueueBuffer = nullptr;
    GPUBuffer *cpCounterBuffer = nullptr;
    GPUBuffer *filteredMeshlets = nullptr;

    std::shared_ptr<GPUBindId> vertexBufferId = nullptr;
    std::shared_ptr<GPUBindId> meshletBufferId = nullptr;
    std::shared_ptr<GPUBindId> meshletVertexBufferId = nullptr;
    std::shared_ptr<GPUBindId> meshletIndexBufferId = nullptr;
    std::shared_ptr<GPUBindId> meshletCullBufferId = nullptr;
    std::shared_ptr<GPUBindId> bvhNodeBufferId = nullptr;
    std::shared_ptr<GPUBindId> clusterGroupBufferId = nullptr;
    std::shared_ptr<GPUBindId> meshletInClusterBufferId = nullptr;
    std::shared_ptr<GPUBindId> cpQueueBufferId = nullptr;
    std::shared_ptr<GPUBindId> cpCounterBufferId = nullptr;
    std::shared_ptr<GPUBindId> filteredMeshletsId = nullptr;

    GPUObjectBuffer objectData;
    GPUBuffer *objectBuffer = nullptr;
    std::shared_ptr<GPUBindId> objectBufferId = nullptr;

  } m_resource;
  bool m_resourceDirty = true;
  std::shared_ptr<MeshData> m_data;

  virtual std::shared_ptr<MeshData> loadMesh() { return m_data; }
  inline void setGPUResource(GPUResource &resource) {
    m_resource.vertexBuffer = resource.vertexBuffer;
    m_resource.meshletBuffer = resource.meshletBuffer;
    m_resource.meshletVertexBuffer = resource.meshletVertexBuffer;
    m_resource.meshletIndexBuffer = resource.meshletIndexBuffer;
    m_resource.meshletCullBuffer = resource.meshletCullBuffer;
    m_resource.bvhNodeBuffer = resource.bvhNodeBuffer;
    m_resource.clusterGroupBuffer = resource.clusterGroupBuffer;
    m_resource.meshletInClusterBuffer = resource.meshletInClusterBuffer;
    m_resource.filteredMeshlets = resource.filteredMeshlets;

    m_resource.vertexBufferId = resource.vertexBufferId;
    m_resource.meshletBufferId = resource.meshletBufferId;
    m_resource.meshletVertexBufferId = resource.meshletVertexBufferId;
    m_resource.meshletIndexBufferId = resource.meshletIndexBufferId;
    m_resource.meshletCullBufferId = resource.meshletCullBufferId;
    m_resource.bvhNodeBufferId = resource.bvhNodeBufferId;
    m_resource.clusterGroupBufferId = resource.clusterGroupBufferId;
    m_resource.meshletInClusterBufferId = resource.meshletInClusterBufferId;
    m_resource.cpQueueBufferId = resource.cpQueueBufferId;

    m_resource.objectBuffer = resource.objectBuffer;
    m_resource.objectBufferId = resource.objectBufferId;
    m_resource.objectData = resource.objectData;
  }
  inline void getGPUResource(GPUResource &resource) {
    resource.vertexBuffer = m_resource.vertexBuffer;
    resource.meshletBuffer = m_resource.meshletBuffer;
    resource.meshletVertexBuffer = m_resource.meshletVertexBuffer;
    resource.meshletIndexBuffer = m_resource.meshletIndexBuffer;
    resource.meshletCullBuffer = m_resource.meshletCullBuffer;
    resource.bvhNodeBuffer = m_resource.bvhNodeBuffer;
    resource.clusterGroupBuffer = m_resource.clusterGroupBuffer;
    resource.meshletInClusterBuffer = m_resource.meshletInClusterBuffer;
    resource.cpQueueBuffer = m_resource.cpQueueBuffer;

    resource.vertexBufferId = m_resource.vertexBufferId;
    resource.meshletBufferId = m_resource.meshletBufferId;
    resource.meshletVertexBufferId = m_resource.meshletVertexBufferId;
    resource.meshletIndexBufferId = m_resource.meshletIndexBufferId;
    resource.meshletCullBufferId = m_resource.meshletCullBufferId;
    resource.bvhNodeBufferId = m_resource.bvhNodeBufferId;
    resource.clusterGroupBufferId = m_resource.clusterGroupBufferId;
    resource.meshletInClusterBufferId = m_resource.meshletInClusterBufferId;
    resource.cpQueueBufferId = m_resource.cpQueueBufferId;

    resource.objectBuffer = m_resource.objectBuffer;
    resource.objectBufferId = m_resource.objectBufferId;
    resource.objectData = m_resource.objectData;
  }
  // TODO: static method
  virtual void createMeshLodHierarchy(std::shared_ptr<MeshData> meshData);

  IFRIT_STRUCT_SERIALIZE(m_data, m_assetReference, m_usingAsset);
};

class MeshFilter : public Component {
private:
  bool m_meshLoaded = false;
  std::shared_ptr<Mesh> m_rawData = nullptr;
  AssetReference m_meshReference;
  std::shared_ptr<Mesh> m_attribute =
      nullptr; // this points to the actual object used
               // for primitive gathering

public:
  MeshFilter() {}
  MeshFilter(std::shared_ptr<SceneObject> owner) : Component(owner) {}
  virtual ~MeshFilter() = default;
  inline std::string serialize() override { return ""; }
  inline void deserialize() override {}
  void loadMesh();
  inline void setMesh(std::shared_ptr<Mesh> p) {
    m_meshReference = p->m_assetReference;
    if (!p->m_usingAsset) {
      m_rawData = p;
    }
    m_attribute = p;
  }
  inline virtual std::vector<AssetReference *> getAssetReferences() override {
    if (m_meshReference.m_usingAsset == false)
      return {};
    return {&m_meshReference};
  }
  inline virtual void setAssetReferencedAttributes(
      const std::vector<std::shared_ptr<IAssetCompatible>> &out) override {
    if (m_meshReference.m_usingAsset) {
      auto mesh = Ifrit::Common::Utility::checked_pointer_cast<Mesh>(out[0]);
      m_attribute = mesh;
    }
  }
  inline std::shared_ptr<Mesh> getMesh() { return m_attribute; }
  IFRIT_COMPONENT_SERIALIZE(m_rawData, m_meshReference);
};

class MeshRenderer : public Component {
private:
  std::shared_ptr<Material> m_material;
  AssetReference m_materialReference;

public:
  MeshRenderer() {} // for deserialization
  MeshRenderer(std::shared_ptr<SceneObject> owner) : Component(owner) {}
  virtual ~MeshRenderer() = default;
  inline std::string serialize() override { return ""; }
  inline void deserialize() override {}
  inline std::shared_ptr<Material> getMaterial() { return m_material; }
  inline void setMaterial(std::shared_ptr<Material> p) { m_material = p; }

  IFRIT_COMPONENT_SERIALIZE(m_materialReference);
};

} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::MeshFilter);
IFRIT_COMPONENT_REGISTER(Ifrit::Core::MeshRenderer);