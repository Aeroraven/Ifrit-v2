
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

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/runtime/assetmanager/Asset.h"
#include "ifrit/runtime/base/Scene.h"
#include <filesystem>
#include <stdexcept>

namespace Ifrit::Runtime
{
    IF_CONSTEXPR const char* cSceneFileExtension = ".scene";

    class SceneAssetManager;

    class IFRIT_APIDECL SceneAsset : public Asset
    {
    public:
        Ref<Scene> m_scene;
        SceneAsset(AssetMetadata metadata, std::filesystem::path path) : Asset(metadata, path) {}
        inline Ref<Scene> GetScene() { return m_scene; }
    };

    class IFRIT_APIDECL SceneAssetImporter : public AssetImporter
    {
    protected:
        SceneAssetManager* m_sceneAssetManager;

    public:
        IF_CONSTEXPR static const char* IMPORTER_NAME = "SceneImporter";
        SceneAssetImporter(AssetManager* manager, SceneAssetManager* sceneManager)
            : AssetImporter(manager), m_sceneAssetManager(sceneManager)
        {
        }
        void        ProcessMetadata(AssetMetadata& metadata) override;
        void        ImportAsset(const std::filesystem::path& path, AssetMetadata& metadata) override;
        Vec<String> GetSupportedExtensionNames() override;
    };

    class IFRIT_APIDECL SceneAssetManager
    {
    private:
        Ref<SceneAssetImporter> m_sceneImporter;
        Vec<Ref<Scene>>         m_scenes;
        Vec<u32>                m_sceneAssetLoaded;
        HashMap<String, u32>    m_scenesIndex;
        Ref<Scene>              m_activeScene;
        std::filesystem::path   m_sceneDataPath;
        AssetManager*           m_assetManager;

    private:
        void AttachAssetResources(Ref<Scene>& scene);

    public:
        SceneAssetManager(std::filesystem::path path, AssetManager* assetManager);
        void                           SaveScenes();
        void                           LoadScenes();
        void                           RegisterScene(String name, Ref<Scene> scene);
        Ref<Scene>                     CreateScene(String name);
        inline Ref<SceneAssetImporter> GetImporter() { return m_sceneImporter; }
        inline Ref<Scene>              GetScene(String name)
        {
            if (m_scenesIndex.count(name) == 0)
            {
                throw std::runtime_error("Scene does not exist");
            }
            if (m_sceneAssetLoaded[m_scenesIndex[name]] == 0)
            {
                AttachAssetResources(m_scenes[m_scenesIndex[name]]);
            }
            return m_scenes[m_scenesIndex[name]];
        }
        inline bool CheckIfSceneExists(String name) { return m_scenesIndex.count(name) != 0; }
    };
} // namespace Ifrit::Runtime
