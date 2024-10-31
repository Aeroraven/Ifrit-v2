#include "ifrit/common/util/Identifier.h"
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/assetmanager/ShaderAsset.h"
#include "ifrit/core/assetmanager/WaveFrontAsset.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace Ifrit::Core {
IFRIT_APIDECL void AssetManager::loadAsset(const std::filesystem::path &path) {
  //std::cout << "Loading asset: " << path << std::endl;
  // Check if this file has a metadata file, add '.meta' without changing suffix
  // name
  auto metaPath = path;
  metaPath += cMetadataFileExtension;
  if (std::filesystem::exists(metaPath)) {
    //std::cout << "Metadata file found: " << metaPath << std::endl;
  } else {
    // No metadata file found, create one

    auto relativePath = std::filesystem::relative(path, basePath);
    AssetMetadata metaData;
    metaData.m_fileId = relativePath.generic_string();
    metaData.m_name = path.filename().generic_string();
    Common::Utility::generateUuid(metaData.m_uuid);
    // check if importer is registered for this file extension
    if (m_extensionImporterMap.find(path.extension().generic_string()) ==
        m_extensionImporterMap.end()) {
      //printf("No importer found \n");
      return;
    }
    auto importerName =
        m_extensionImporterMap[path.extension().generic_string()];
    auto importer = m_importers[importerName];
    importer->processMetadata(metaData);
    std::string serialized;
    serialized = metadataSerialization(metaData);
    std::ofstream file(metaPath);
    file << serialized;
    file.close();
  }

  // Deserialize metadata and import asset
  std::ifstream file(metaPath);
  std::string serialized;
  file.seekg(0, std::ios::end);
  serialized.reserve(file.tellg());
  file.seekg(0, std::ios::beg);
  serialized.assign((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
  AssetMetadata metadata;
  metadataDeserialization(serialized, metadata);
  auto importerName = metadata.m_importer;
  // check if importer is registered
  if (m_importers.find(importerName) == m_importers.end()) {
    printf("Importer not found: %s\n", importerName.c_str());
    return;
  }
  auto importer = m_importers[importerName];
  importer->importAsset(path, metadata);
}
IFRIT_APIDECL void
AssetManager::loadAssetDirectory(const std::filesystem::path &path) {
  if (!std::filesystem::exists(path)) {
    auto s = path.generic_string();
    printf("Path does not exist: %s\n", s.c_str());
  }
  for (auto &entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_directory()) {
      loadAssetDirectory(entry.path());
    } else {
      // check if this is a metadata file, if so, ignore it
      if (entry.path().extension() == ".meta") {
        continue;
      }
      // load the asset
      loadAsset(entry.path());
    }
  }
}

IFRIT_APIDECL void
AssetManager::registerImporter(const std::string &importerName,
                               std::shared_ptr<AssetImporter> importer) {
  m_importers[importerName] = importer;
  auto extensions = importer->getSupportedExtensionNames();
  for (auto &ext : extensions) {
    m_extensionImporterMap[ext] = importerName;
  }
}

IFRIT_APIDECL AssetManager::AssetManager(std::filesystem::path path) {
  // register default importers
  // TODO: maybe weak_ptr should be used, but i am too lazy to do that
  registerImporter(WaveFrontAssetImporter::IMPORTER_NAME,
                   std::make_shared<WaveFrontAssetImporter>(this));
  registerImporter(ShaderAssetImporter::IMPORTER_NAME,
                   std::make_shared<ShaderAssetImporter>(this));
  basePath = path;
  // loadAssetDirectory(basePath);
}

IFRIT_APIDECL std::string
AssetManager::metadataSerialization(AssetMetadata &metadata) {
  std::string serialized;
  Ifrit::Common::Serialization::serialize(metadata, serialized);
  return serialized;
}

IFRIT_APIDECL void
AssetManager::metadataDeserialization(const std::string &serialized,
                                      AssetMetadata &metadata) {
  Ifrit::Common::Serialization::deserialize(serialized, metadata);
}

IFRIT_APIDECL void AssetManager::registerAsset(std::shared_ptr<Asset> asset) {
  if (m_assets.find(asset->getUuid()) != m_assets.end()) {
    throw std::runtime_error("Asset already registered");
  }
  m_assets[asset->getUuid()] = asset;
  m_nameToUuid[asset->getFileId()] = asset->getUuid();
}
} // namespace Ifrit::Core
