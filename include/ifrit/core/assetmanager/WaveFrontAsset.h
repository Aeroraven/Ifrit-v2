#pragma once
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/Mesh.h"

namespace Ifrit::Core {
class IFRIT_APIDECL WaveFrontAsset : public Asset, public Mesh {
private:
  std::shared_ptr<MeshData> m_selfData;
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