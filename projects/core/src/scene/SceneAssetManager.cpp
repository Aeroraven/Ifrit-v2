
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

#include "ifrit/common/logging/Logging.h"

#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/scene/SceneAssetManager.h"
#include <fstream>

using namespace Ifrit::Common::Utility;

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
  m_sceneAssetManager->registerScene(fileName, asset->getScene());

  iInfo("Imported asset: [Scene] {}", metadata.m_uuid);
}

// Manager
IFRIT_APIDECL void
SceneAssetManager::attachAssetResources(std::shared_ptr<Scene> &scene) {
  std::vector<std::shared_ptr<Component>> components;
  std::vector<SceneNode *> nodes;
  nodes.push_back(scene->getRootNode().get());
  while (!nodes.empty()) {
    auto node = nodes.back();
    nodes.pop_back();
    auto children = node->getChildren();
    for (const auto &child : children) {
      nodes.push_back(child.get());
    }
    for (auto &obj : node->getGameObjects()) {
      for (auto &x : obj->getAllComponents()) {
        components.push_back(x);
      }
    }
  }
  for (auto &x : components) {
    auto config = x->getAssetReferences();
    std::vector<std::shared_ptr<IAssetCompatible>> assets;
    for (auto &y : config) {
      auto asset = m_assetManager->getAssetByName<Asset>(y->m_name);
      if (asset == nullptr) {
        throw std::runtime_error("Asset not found");
      }
      assets.push_back(asset);
    }
    x->setAssetReferencedAttributes(assets);
  }
}

IFRIT_APIDECL SceneAssetManager::SceneAssetManager(std::filesystem::path path,
                                                   AssetManager *assetman)
    : m_sceneDataPath(path), m_assetManager(assetman) {
  m_sceneImporter = std::make_shared<SceneAssetImporter>(assetman, this);
  assetman->registerImporter(m_sceneImporter->IMPORTER_NAME, m_sceneImporter);
}

IFRIT_APIDECL void SceneAssetManager::saveScenes() {
  using namespace Ifrit::Common::Serialization;
  for (auto &[name, idx] : m_scenesIndex) {
    auto scene = m_scenes[idx];
    std::string serialized;
    serialize(scene, serialized);
    auto fileName = m_sceneDataPath / (name + cSceneFileExtension);
    std::ofstream file(fileName);
    file << serialized;
    file.close();
  }
}

IFRIT_APIDECL void SceneAssetManager::loadScenes() {
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
    // m_scenes[name] = scene;
    m_scenesIndex[name] = size_cast<uint32_t>(m_scenes.size());
    m_scenes.push_back(scene);
  }
}

IFRIT_APIDECL std::shared_ptr<Scene>
SceneAssetManager::createScene(std::string name) {
  auto scene = std::make_shared<Scene>();
  // m_scenes[name] = scene;
  m_scenesIndex[name] = size_cast<uint32_t>(m_scenes.size());
  m_scenes.push_back(scene);
  m_sceneAssetLoaded.push_back(1);
  return scene;
}

IFRIT_APIDECL void
SceneAssetManager::registerScene(std::string name,
                                 std::shared_ptr<Scene> scene) {
  // m_scenes[name] = scene;
  m_scenesIndex[name] = size_cast<uint32_t>(m_scenes.size());
  m_scenes.push_back(scene);
  m_sceneAssetLoaded.push_back(0);
}
} // namespace Ifrit::Core