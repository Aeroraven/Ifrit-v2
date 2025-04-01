
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

#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/algo/Identifier.h"
#include "ifrit/runtime/assetmanager/Asset.h"
#include "ifrit/runtime/assetmanager/DirectDrawSurfaceAsset.h"
#include "ifrit/runtime/assetmanager/GLTFAsset.h"
#include "ifrit/runtime/assetmanager/ShaderAsset.h"
#include "ifrit/runtime/assetmanager/TrivialImageAsset.h"
#include "ifrit/runtime/assetmanager/WaveFrontAsset.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace Ifrit::Runtime
{
    IFRIT_APIDECL void AssetManager::LoadAsset(const std::filesystem::path& path)
    {
        // std::cout << "Loading asset: " << path << std::endl;
        //  Check if this file has a metadata file, add '.meta' without changing
        //  suffix name
        auto metaPath = path;
        metaPath += cMetadataFileExtension;
        auto relativePath = std::filesystem::relative(path, basePath);
        if (m_nameToUuid.find(relativePath.generic_string()) != m_nameToUuid.end())
        {
            iWarn("Asset already loaded: {}", path.generic_string());
            return;
        }
        if (std::filesystem::exists(metaPath))
        {
            // std::cout << "Metadata file found: " << metaPath << std::endl;
        }
        else
        {
            // No metadata file found, create one
            AssetMetadata metaData;
            metaData.m_fileId = relativePath.generic_string();
            metaData.m_name   = path.filename().generic_string();
            GenerateUuid(metaData.m_uuid);
            // check if importer is registered for this file extension
            if (m_extensionImporterMap.find(path.extension().generic_string()) == m_extensionImporterMap.end())
            {
                iWarn("No importer found for file: {}", path.generic_string());
                return;
            }
            auto importerName = m_extensionImporterMap[path.extension().generic_string()];
            auto importer     = m_importers[importerName];
            importer->ProcessMetadata(metaData);
            String serialized;
            serialized = MetadataSerialization(metaData);
            std::ofstream file(metaPath);
            file << serialized;
            file.close();
        }

        // Deserialize metadata and import asset
        std::ifstream file(metaPath);
        String        serialized;
        file.seekg(0, std::ios::end);
        serialized.reserve(file.tellg());
        file.seekg(0, std::ios::beg);
        serialized.assign((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        AssetMetadata metadata;
        MetadataDeserialization(serialized, metadata);
        auto importerName = metadata.m_importer;
        // check if importer is registered
        if (m_importers.find(importerName) == m_importers.end())
        {
            iWarn("Importer not found: {}", importerName);
            return;
        }
        auto importer = m_importers[importerName];
        importer->ImportAsset(path, metadata);
    }

    IFRIT_APIDECL void AssetManager::LoadAssetDirectory(const std::filesystem::path& path)
    {
        if (!std::filesystem::exists(path))
        {
            auto s = path.generic_string();
            iWarn("Path does not exist: {}", s);
        }
        for (auto& entry : std::filesystem::directory_iterator(path))
        {
            if (entry.is_directory())
            {
                LoadAssetDirectory(entry.path());
            }
            else
            {
                // check if this is a metadata file, if so, ignore it
                if (entry.path().extension() == ".meta")
                {
                    continue;
                }
                // load the asset
                LoadAsset(entry.path());
            }
        }
    }

    IFRIT_APIDECL std::shared_ptr<Asset> AssetManager::RequestAssetIntenal(const std::filesystem::path& path)
    {
        auto relativePath = std::filesystem::relative(path, basePath);
        if (m_nameToUuid.find(relativePath.generic_string()) == m_nameToUuid.end())
        {
            // load the asset
            LoadAsset(path);
            if (m_nameToUuid.find(relativePath.generic_string()) == m_nameToUuid.end())
            {
                iError("Asset not found: {}", path.generic_string());
                return nullptr;
            }
        }
        auto uuid = m_nameToUuid[relativePath.generic_string()];
        auto it   = m_assets.find(uuid);
        if (it == m_assets.end())
        {
            iWarn("Asset not found: {}", path.generic_string());
            return nullptr;
        }
        return it->second;
    }

    IFRIT_APIDECL void AssetManager::RegisterImporter(
        const String& importerName, std::shared_ptr<AssetImporter> importer)
    {
        m_importers[importerName] = importer;
        auto extensions           = importer->GetSupportedExtensionNames();
        for (auto& ext : extensions)
        {
            m_extensionImporterMap[ext] = importerName;
        }
    }

    IFRIT_APIDECL AssetManager::AssetManager(std::filesystem::path path, IApplication* app)
    {
        // register default importers
        // TODO: maybe weak_ptr should be used, but i am too lazy to do that
        RegisterImporter(WaveFrontAssetImporter::IMPORTER_NAME, std::make_shared<WaveFrontAssetImporter>(this));
        RegisterImporter(ShaderAssetImporter::IMPORTER_NAME, std::make_shared<ShaderAssetImporter>(this));
        RegisterImporter(GLTFAssetImporter::IMPORTER_NAME, std::make_shared<GLTFAssetImporter>(this));
        RegisterImporter(
            DirectDrawSurfaceAssetImporter::IMPORTER_NAME, std::make_shared<DirectDrawSurfaceAssetImporter>(this));
        RegisterImporter(TrivialImageAssetImporter::IMPORTER_NAME, std::make_shared<TrivialImageAssetImporter>(this));
        basePath = path;
        m_app    = app;
        // LoadAssetDirectory(basePath);
    }

    IFRIT_APIDECL String AssetManager::MetadataSerialization(AssetMetadata& metadata)
    {
        String serialized;
        Ifrit::Common::Serialization::serialize(metadata, serialized);
        return serialized;
    }

    IFRIT_APIDECL void AssetManager::MetadataDeserialization(const String& serialized, AssetMetadata& metadata)
    {
        Ifrit::Common::Serialization::deserialize(serialized, metadata);
    }

    IFRIT_APIDECL void AssetManager::RegisterAsset(std::shared_ptr<Asset> asset)
    {
        if (m_assets.find(asset->getUuid()) != m_assets.end())
        {
            throw std::runtime_error("Asset already registered");
        }
        m_assets[asset->getUuid()]       = asset;
        m_nameToUuid[asset->getFileId()] = asset->getUuid();
    }
} // namespace Ifrit::Runtime
