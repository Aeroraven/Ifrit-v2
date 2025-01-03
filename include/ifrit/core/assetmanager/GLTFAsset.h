
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
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/Mesh.h"

namespace Ifrit::Core {
class GLTFAsset;
struct GLTFInternalData;
class GLTFAssetImporter;

class IFRIT_APIDECL GLTFMesh : public Mesh {
private:
  GLTFAsset *m_asset;
  uint32_t m_meshId;
  uint32_t m_primitiveId;
  uint32_t m_nodeId;
  std::shared_ptr<MeshData> m_selfData;
  MeshData *m_selfDataRaw = nullptr;
  bool m_loaded = false;
  std::string m_cachePath;

public:
  GLTFMesh(AssetMetadata *metadata, GLTFAsset *asset, uint32_t meshId,
           uint32_t primitiveId, uint32_t nodeId, const std::string &cachePath)
      : m_asset(asset), m_meshId(meshId), m_primitiveId(primitiveId),
        m_nodeId(nodeId), m_cachePath(cachePath) {
    m_assetReference.m_fileId = metadata->m_fileId;
    m_assetReference.m_name = metadata->m_name;
    m_assetReference.m_uuid = metadata->m_uuid;
    m_assetReference.m_usingAsset = true;
    m_usingAsset = true;
  }

  virtual std::shared_ptr<MeshData> loadMesh() override;
  virtual MeshData *loadMeshUnsafe() override;
};

class IFRIT_APIDECL GLTFPrefab {
public:
  GLTFAsset *m_asset;
  uint32_t m_meshId;
  uint32_t m_primitiveId;
  uint32_t m_nodeId;
  std::shared_ptr<SceneObjectPrefab> m_prefab;
  GLTFPrefab(AssetMetadata *metadata, GLTFAsset *asset, uint32_t meshId,
             uint32_t primitiveId, uint32_t nodeId,
             const float4x4 &parentTransform);
};

class IFRIT_APIDECL GLTFAsset : public Asset {
private:
  bool m_loaded = false;
  AssetMetadata m_metadata;
  std::filesystem::path m_path;
  std::vector<std::shared_ptr<GLTFMesh>> m_meshes;
  std::vector<std::shared_ptr<GLTFPrefab>> m_prefabs;
  GLTFInternalData *m_internalData = nullptr;

private:
  inline AssetMetadata &getMetadata() { return m_metadata; }

public:
  GLTFAsset(AssetMetadata metadata, std::filesystem::path path,
            AssetManager *m_manager)
      : Asset(metadata, path), m_metadata(metadata), m_path(path) {
    loadGLTF(m_manager);
  }
  ~GLTFAsset();
  void loadGLTF(AssetManager *m_manager);
  GLTFInternalData *getInternalData();
  inline std::vector<std::shared_ptr<GLTFPrefab>> getPrefabs() {
    return m_prefabs;
  }

  friend class GLTFMesh;
  friend class GLTFPrefab;
};

class IFRIT_APIDECL GLTFAssetImporter : public AssetImporter {
public:
  constexpr static const char *IMPORTER_NAME = "GLTFAssetImporter";
  GLTFAssetImporter(AssetManager *manager) : AssetImporter(manager) {}
  void processMetadata(AssetMetadata &metadata) override;
  void importAsset(const std::filesystem::path &path,
                   AssetMetadata &metadata) override;
  std::vector<std::string> getSupportedExtensionNames() override;
};
} // namespace Ifrit::Core