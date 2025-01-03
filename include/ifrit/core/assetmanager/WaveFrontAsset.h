
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
class IFRIT_APIDECL WaveFrontAsset : public Asset, public Mesh {
private:
  std::shared_ptr<MeshData> m_selfData;
  MeshData *m_selfDataRaw = nullptr;
  bool m_loaded = false;

public:
  WaveFrontAsset(AssetMetadata metadata, std::filesystem::path path)
      : Asset(metadata, path) {
    m_assetReference.m_fileId = metadata.m_fileId;
    m_assetReference.m_name = metadata.m_name;
    m_assetReference.m_uuid = metadata.m_uuid;
    m_assetReference.m_usingAsset = true;
    m_usingAsset = true;
  }
  std::shared_ptr<MeshData> loadMesh() override;
  MeshData *loadMeshUnsafe() override;
  inline Mesh &getMesh() { return *this; }
};
class IFRIT_APIDECL WaveFrontAssetImporter : public AssetImporter {
public:
  constexpr static const char *IMPORTER_NAME = "WaveFrontImporter";
  WaveFrontAssetImporter(AssetManager *manager) : AssetImporter(manager) {}
  void processMetadata(AssetMetadata &metadata) override;
  void importAsset(const std::filesystem::path &path,
                   AssetMetadata &metadata) override;
  std::vector<std::string> getSupportedExtensionNames() override;
};
} // namespace Ifrit::Core