
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

#include "ifrit/runtime/scene/SceneAssetManager.h"
#include "ifrit/core/serialization/Serializer.h"
#include <fstream>

using namespace Ifrit;

namespace Ifrit::Runtime
{

    // Manager
    IFRIT_APIDECL void SceneAssetManager::AttachAssetResources(Ref<Scene>& scene)
    {
       
    }

    IFRIT_APIDECL SceneAssetManager::SceneAssetManager(std::filesystem::path path, AssetManager* assetman)
        : m_sceneDataPath(path), m_assetManager(assetman)
    {

    }

    IFRIT_APIDECL void SceneAssetManager::SaveScenes()
    {
        IF_LOG_ERROR("SceneAssetImporter", "Removed function");
        using namespace Ifrit::Serialization;
        for (auto& [name, idx] : m_scenesIndex)
        {
            auto   scene = m_scenes[idx];
            String serialized;
            //SerializeBinary(scene, serialized);
            auto          fileName = m_sceneDataPath / (name + cSceneFileExtension);
            std::ofstream file(fileName);
            file << serialized;
            file.close();
        }
    }

    IFRIT_APIDECL void SceneAssetManager::LoadScenes()
    {
        IF_LOG_ERROR("SceneAssetImporter", "Removed function");
        using namespace Ifrit::Serialization;
        for (auto& entry : std::filesystem::directory_iterator(m_sceneDataPath))
        {
            if (entry.is_directory())
            {
                continue;
            }
            if (entry.path().extension() != cSceneFileExtension)
            {
                continue;
            }
            std::ifstream file(entry.path());
            String        serialized;
            file.seekg(0, std::ios::end);
            serialized.reserve(file.tellg());
            file.seekg(0, std::ios::beg);
            serialized.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            Ref<Scene> scene;
            // use the name of the file as the key, extension removed
            auto name = entry.path().filename().replace_extension("").generic_string();
            // m_scenes[name] = scene;
            m_scenesIndex[name] = SizeCast<uint32_t>(m_scenes.size());
            m_scenes.push_back(scene);
        }
    }

    IFRIT_APIDECL Ref<Scene> SceneAssetManager::CreateScene(String name)
    {
        auto scene = MakeRef<Scene>();
        // m_scenes[name] = scene;
        m_scenesIndex[name] = SizeCast<uint32_t>(m_scenes.size());
        m_scenes.push_back(scene);
        m_sceneAssetLoaded.push_back(1);
        return scene;
    }

    IFRIT_APIDECL void SceneAssetManager::RegisterScene(String name, Ref<Scene> scene)
    {
        // m_scenes[name] = scene;
        m_scenesIndex[name] = SizeCast<uint32_t>(m_scenes.size());
        m_scenes.push_back(scene);
        m_sceneAssetLoaded.push_back(0);
    }
} // namespace Ifrit::Runtime