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
  std::filesystem::path m_path;

public:
  Asset(AssetMetadata metadata, std::filesystem::path path)
      : m_metadata(metadata), m_path(path) {}

  const std::string &getUuid() const { return m_metadata.m_uuid; }
  const std::string &getName() const { return m_metadata.m_name; }
  const std::string &getFileId() const { return m_metadata.m_fileId; }
  virtual void _polyHolder(){}
};

class IFRIT_APIDECL AssetImporter {
protected:
  AssetManager *m_assetManager = nullptr;

public:
  AssetImporter(AssetManager *manager) : m_assetManager(manager) {}
  virtual AssetMetadata
  requestDependencyMeta(const std::filesystem::path &path) {
    return AssetMetadata();
  }
  virtual void importAsset(const std::filesystem::path &path,
                           AssetMetadata &metadata) = 0;
  virtual void processMetadata(AssetMetadata &metadata) = 0;
  virtual std::vector<std::string> getSupportedExtensionNames() = 0;
};

class IFRIT_APIDECL AssetManager {
private:
  std::unordered_map<std::string, std::shared_ptr<Asset>> m_assets;
  std::unordered_map<std::string, std::string> m_nameToUuid;
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

  inline void loadAssetDirectory() { loadAssetDirectory(basePath); }

  void registerImporter(const std::string &extensionName,
                        std::shared_ptr<AssetImporter> importer);
  void registerAsset(std::shared_ptr<Asset> asset);

  template <typename T> std::shared_ptr<T> getAsset(const std::string &uuid) {
    auto it = m_assets.find(uuid);
    if (it == m_assets.end()) {
      return nullptr;
    }
    return std::dynamic_pointer_cast<T>(it->second);
  }

  template <typename T>
  std::shared_ptr<T> getAssetByName(const std::string &name) {
    auto it = m_nameToUuid.find(name);
    if (it == m_nameToUuid.end()) {
      return nullptr;
    }
    return getAsset<T>(it->second);
  }
};
} // namespace Ifrit::Core
