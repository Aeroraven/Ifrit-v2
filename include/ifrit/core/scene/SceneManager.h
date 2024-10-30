#pragma once
#include "ifrit/core/assetmanager/Asset.h"
#include "ifrit/core/base/Scene.h"
#include <filesystem>

namespace Ifrit::Core {
constexpr const char *cSceneFileExtension = ".scene";

class SceneManager;

class IFRIT_APIDECL SceneAsset : public Asset {
public:
  std::shared_ptr<Scene> m_scene;
  SceneAsset(AssetMetadata metadata, std::filesystem::path path)
      : Asset(metadata, path) {}
  inline std::shared_ptr<Scene> getScene() { return m_scene; }
};

class IFRIT_APIDECL SceneAssetImporter : public AssetImporter {
protected:
  SceneManager *m_sceneManager;

public:
  constexpr static const char *IMPORTER_NAME = "SceneImporter";
  SceneAssetImporter(AssetManager *manager, SceneManager *sceneManager)
      : AssetImporter(manager), m_sceneManager(sceneManager) {}
  void processMetadata(AssetMetadata &metadata) override;
  void importAsset(const std::filesystem::path &path,
                   AssetMetadata &metadata) override;
  std::vector<std::string> getSupportedExtensionNames() override;
};

class IFRIT_APIDECL SceneManager {
private:
  std::shared_ptr<SceneAssetImporter> m_sceneImporter;
  std::unordered_map<std::string, std::shared_ptr<Scene>> m_scenes;
  std::shared_ptr<Scene> m_activeScene;
  std::filesystem::path m_sceneDataPath;

public:
  SceneManager(std::filesystem::path path, AssetManager *assetManager);
  void saveScenes();
  void loadScenes();
  void registerScene(std::string name, std::shared_ptr<Scene> scene);
  std::shared_ptr<Scene> createScene(std::string name);
  inline std::shared_ptr<SceneAssetImporter> getImporter() {
    return m_sceneImporter;
  }
  inline std::shared_ptr<Scene> getScene(std::string name) {
    if (m_scenes.count(name) == 0) {
      throw std::exception("not found");
    }
    return m_scenes[name];
  }
  inline bool checkSceneExists(std::string name) {
    return m_scenes.count(name) != 0;
  }
};
} // namespace Ifrit::Core
