#include "ifrit/core/scene/SceneManager.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include <fstream>

namespace Ifrit::Core {
// Importer
IFRIT_APIDECL void
SceneAssetImporter::processMetadata(AssetMetadata &metadata) {
  metadata.m_importer = IMPORTER_NAME;
};

IFRIT_APIDECL std::vector<std::string>
SceneAssetImporter::getSupportedExtensionNames() {
  return {cSceneFileExtension};
};

IFRIT_APIDECL void
SceneAssetImporter::importAsset(const std::filesystem::path &path,
                                AssetMetadata &metadata) {
  auto asset = std::make_shared<SceneAsset>(metadata, path);
  std::string fileReaded;
  std::ifstream file(path);
  file.seekg(0, std::ios::end);
  fileReaded.reserve(file.tellg());
  file.seekg(0, std::ios::beg);
  fileReaded.assign((std::istreambuf_iterator<char>(file)),
                    std::istreambuf_iterator<char>());
  file.close();
  std::shared_ptr<Scene> x;
  Ifrit::Common::Serialization::deserialize(fileReaded, x);
  asset->m_scene = x;
  auto fileName = metadata.m_name;
  // remove extension
  fileName = fileName.substr(0, fileName.find_last_of("."));
  m_assetManager->registerAsset(asset);
  m_sceneManager->registerScene(fileName, asset->getScene());
  printf("Imported asset: [Scene] %s\n", metadata.m_uuid.c_str());
}

// Manager
IFRIT_APIDECL SceneManager::SceneManager(std::filesystem::path path,
                                         AssetManager *assetman)
    : m_sceneDataPath(path) {
  m_sceneImporter = std::make_shared<SceneAssetImporter>(assetman, this);
  assetman->registerImporter(m_sceneImporter->IMPORTER_NAME, m_sceneImporter);
}

IFRIT_APIDECL void SceneManager::saveScenes() {
  using namespace Ifrit::Common::Serialization;
  for (auto &[name, scene] : m_scenes) {
    std::string serialized;
    serialize(scene, serialized);
    auto fileName = m_sceneDataPath / (name + cSceneFileExtension);
    std::ofstream file(fileName);
    file << serialized;
    file.close();
  }
}

IFRIT_APIDECL void SceneManager::loadScenes() {
  using namespace Ifrit::Common::Serialization;
  for (auto &entry : std::filesystem::directory_iterator(m_sceneDataPath)) {
    if (entry.is_directory()) {
      continue;
    }
    if (entry.path().extension() != cSceneFileExtension) {
      continue;
    }
    std::ifstream file(entry.path());
    std::string serialized;
    file.seekg(0, std::ios::end);
    serialized.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    serialized.assign((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    std::shared_ptr<Scene> scene;
    deserialize(serialized, scene);
    // use the name of the file as the key, extension removed
    auto name = entry.path().filename().replace_extension("").generic_string();
    m_scenes[name] = scene;
  }
}

IFRIT_APIDECL std::shared_ptr<Scene>
SceneManager::createScene(std::string name) {
  auto scene = std::make_shared<Scene>();
  m_scenes[name] = scene;
  return scene;
}

IFRIT_APIDECL void SceneManager::registerScene(std::string name,
                                               std::shared_ptr<Scene> scene) {
  m_scenes[name] = scene;
}
} // namespace Ifrit::Core