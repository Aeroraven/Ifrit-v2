#include "ifrit/core/assetmanager/ShaderAsset.h"
#include <fstream>
namespace Ifrit::Core {

// Shader class
IFRIT_APIDECL std::vector<char> ShaderAsset::loadShader() {
  if (m_loaded) {
    return *m_selfData;
  } else {
    m_loaded = true;
    std::ifstream file(m_path, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    m_selfData = std::make_shared<std::vector<char>>(data);
    return data;
  }
}

// Importer
IFRIT_APIDECL void
ShaderAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
}

IFRIT_APIDECL std::vector<std::string>
ShaderAssetImporter::getSupportedExtensionNames() {
  return {".spv"};
}

IFRIT_APIDECL void
ShaderAssetImporter::importAsset(const std::filesystem::path &path,
                                 AssetMetadata &metadata) {
  auto asset = std::make_shared<ShaderAsset>(metadata, path);
  m_assetManager->registerAsset(asset);
  printf("Imported asset: [Shader] %s\n", metadata.m_uuid.c_str());
}

} // namespace Ifrit::Core