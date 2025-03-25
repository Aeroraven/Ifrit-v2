
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
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/Mesh.h"

namespace Ifrit::Core {
class GLTFAsset;
struct GLTFInternalData;
class GLTFAssetImporter;

class IFRIT_APIDECL GLTFMesh : public Mesh {
private:
  GLTFAsset *m_asset;
  u32 m_meshId;
  u32 m_primitiveId;
  u32 m_nodeId;
  Ref<MeshData> m_selfData;
  MeshData *m_selfDataRaw = nullptr;
  bool m_loaded = false;
  String m_cachePath;

public:
  GLTFMesh(AssetMetadata *metadata, GLTFAsset *asset, u32 meshId, u32 primitiveId, u32 nodeId, const String &cachePath)
      : m_asset(asset), m_meshId(meshId), m_primitiveId(primitiveId), m_nodeId(nodeId), m_cachePath(cachePath) {
    m_assetReference.m_fileId = metadata->m_fileId;
    m_assetReference.m_name = metadata->m_name;
    m_assetReference.m_uuid = metadata->m_uuid;
    m_assetReference.m_usingAsset = true;
    m_usingAsset = true;
  }

  virtual Ref<MeshData> loadMesh() override;
  virtual MeshData *loadMeshUnsafe() override;
};

class IFRIT_APIDECL GLTFPrefab {
public:
  GLTFAsset *m_asset;
  u32 m_meshId;
  u32 m_primitiveId;
  u32 m_nodeId;
  Ref<SceneObjectPrefab> m_prefab;
  GLTFPrefab(AssetMetadata *metadata, GLTFAsset *asset, u32 meshId, u32 primitiveId, u32 nodeId,
             const Matrix4x4f &parentTransform);
};

class IFRIT_APIDECL GLTFAsset : public Asset {
private:
  bool m_loaded = false;
  AssetMetadata m_metadata;
  std::filesystem::path m_path;
  Vec<Ref<GLTFMesh>> m_meshes;
  Vec<Ref<GLTFPrefab>> m_prefabs;
  GLTFInternalData *m_internalData = nullptr;

  AssetManager *m_manager;

private:
  inline AssetMetadata &getMetadata() { return m_metadata; }

public:
  GLTFAsset(AssetMetadata metadata, std::filesystem::path path, AssetManager *manager)
      : Asset(metadata, path), m_metadata(metadata), m_path(path) {
    // loadGLTF(m_manager);
    m_manager = manager;
  }
  ~GLTFAsset();
  void loadGLTF(AssetManager *m_manager);
  void requestLoad();
  GLTFInternalData *getInternalData();
  inline Vec<Ref<GLTFPrefab>> getPrefabs() {
    requestLoad();
    return m_prefabs;
  }

  friend class GLTFMesh;
  friend class GLTFPrefab;
};

class IFRIT_APIDECL GLTFAssetImporter : public AssetImporter {
public:
  IF_CONSTEXPR static const char *IMPORTER_NAME = "GLTFAssetImporter";
  GLTFAssetImporter(AssetManager *manager) : AssetImporter(manager) {}
  void processMetadata(AssetMetadata &metadata) override;
  void importAsset(const std::filesystem::path &path, AssetMetadata &metadata) override;
  Vec<String> getSupportedExtensionNames() override;
};
} // namespace Ifrit::Core