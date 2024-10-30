#pragma once
#include "ifrit/core/assetmanager/Asset.h"

namespace Ifrit::Core {
class IFRIT_APIDECL ShaderAsset : public Asset {
private:
  std::shared_ptr<std::vector<char>> m_selfData;
  bool m_loaded = false;

public:
  ShaderAsset(AssetMetadata metadata, std::filesystem::path path)
      : Asset(metadata, path) {}

  std::vector<char> loadShader();
};
class IFRIT_APIDECL ShaderAssetImporter : public AssetImporter {
public:
  constexpr static const char *IMPORTER_NAME = "ShaderImporter";
  ShaderAssetImporter(AssetManager *manager) : AssetImporter(manager) {}
  void processMetadata(AssetMetadata &metadata) override;
  void importAsset(const std::filesystem::path &path,
                   AssetMetadata &metadata) override;
  std::vector<std::string> getSupportedExtensionNames() override;
};
} // namespace Ifrit::Core