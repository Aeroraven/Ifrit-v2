#pragma once
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include <filesystem>
#include <memory>
#include <string>

namespace Ifrit::Core {
constexpr const char *cMetadataFileExtension = ".meta";

class AssetManager;

enum class AssetType {
  Mesh,
  Material,
  Texture,
  Shader,
  Scene,
  Animation,
  Sound,
  Font,
  Script,
  Prefab,
};

// Just a placeholder for now
struct AssetMetadata {
  std::string m_uuid;
  std::string m_name;
  std::string m_fileId;
  std::string m_importer;
  std::vector<std::string> m_dependenciesId;
  std::unordered_map<std::string, std::string> m_importerOptions;
  IFRIT_STRUCT_SERIALIZE(m_uuid, m_name, m_fileId, m_importer,
                         m_importerOptions);
};

class AssetImporter;

class IFRIT_APIDECL Asset {
protected:
  AssetMetadata m_metadata;
};

class IFRIT_APIDECL AssetImporter {
private:
  AssetManager *m_assetManager = nullptr;

public:
  AssetImporter(AssetManager *manager) : m_assetManager(manager) {}
  virtual AssetMetadata
  requestDependencyMeta(const std::filesystem::path &path) {
    return AssetMetadata();
  }
  virtual void processMetadata(AssetMetadata &metadata) = 0;
  virtual std::vector<std::string> getSupportedExtensionNames() = 0;
};

class IFRIT_APIDECL AssetManager {
private:
  std::unordered_map<std::string, std::shared_ptr<Asset>> m_assets;
  std::unordered_map<std::string, std::shared_ptr<AssetImporter>> m_importers;
  std::unordered_map<std::string, std::string> m_extensionImporterMap;
  std::filesystem::path basePath;

private:
  std::string metadataSerialization(AssetMetadata &metadata);
  void metadataDeserialization(const std::string &serialized,
                               AssetMetadata &metadata);

public:
  AssetManager(std::filesystem::path path);
  void loadAsset(const std::filesystem::path &path);
  void loadAssetDirectory(const std::filesystem::path &path);

  void registerImporter(const std::string &extensionName,
                        std::shared_ptr<AssetImporter> importer);
};
} // namespace Ifrit::Core
