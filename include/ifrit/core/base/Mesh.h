#pragma once
#include "AssetReference.h"
#include "Component.h"

namespace Ifrit::Core {
struct MeshData {
  std::vector<ifloat3> m_vertices;
  std::vector<ifloat3> m_normals;
  std::vector<ifloat2> m_uvs;
  std::vector<ifloat3> m_tangents;
  std::vector<uint32_t> m_indices;

  IFRIT_STRUCT_SERIALIZE(m_vertices, m_normals, m_uvs, m_tangents, m_indices);
};

class Mesh {
public:
  std::shared_ptr<MeshData> m_data;
  AssetReference m_assetReference;
  bool m_assetLoaded = false;
  virtual std::shared_ptr<MeshData> loadMesh() { return m_data; }
  IFRIT_STRUCT_SERIALIZE(m_data, m_assetReference, m_assetLoaded);
};

struct Material {
  AssetReference m_assetReference;
  IFRIT_STRUCT_SERIALIZE(m_assetReference);
};

class MeshFilter : public Component, public AttributeOwner<Mesh> {
private:
  std::shared_ptr<MeshData> m_loadedMesh;
  bool m_meshLoaded = false;

public:
  MeshFilter() {}
  MeshFilter(std::shared_ptr<SceneObject> owner)
      : Component(owner), AttributeOwner() {}
  virtual ~MeshFilter() = default;
  inline std::string serialize() override { return serializeAttribute(); }
  inline void deserialize() override { deserializeAttribute(); }
  void loadMesh();
  inline void setMesh(const Mesh& p) { m_attributes = p;  }
  IFRIT_COMPONENT_SERIALIZE(m_attributes);
};

struct MeshRendererData {
  Material m_material;
  IFRIT_STRUCT_SERIALIZE(m_material);
};

class MeshRenderer : public Component, public AttributeOwner<MeshRendererData> {
public:
  MeshRenderer(std::shared_ptr<SceneObject> owner)
      : Component(owner), AttributeOwner() {}
  virtual ~MeshRenderer() = default;
  inline std::string serialize() override { return serializeAttribute(); }
  inline void deserialize() override { deserializeAttribute(); }
};

} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::MeshFilter);