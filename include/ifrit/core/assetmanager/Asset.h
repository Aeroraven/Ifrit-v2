
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
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/base/AssetReference.h"
#include <filesystem>
#include <memory>
#include <string>

namespace Ifrit::Core
{
    IF_CONSTEXPR const char* cMetadataFileExtension = ".meta";

    class AssetManager;

    enum class AssetType
    {
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
    struct AssetMetadata
    {
        String                  m_uuid;
        String                  m_name;
        String                  m_fileId;
        String                  m_importer;
        Vec<String>             m_dependenciesId;
        HashMap<String, String> m_importerOptions;
        IFRIT_STRUCT_SERIALIZE(m_uuid, m_name, m_fileId, m_importer, m_importerOptions);
    };

    class AssetImporter;

    class IFRIT_APIDECL Asset : public std::enable_shared_from_this<Asset>, public IAssetCompatible
    {
    protected:
        AssetMetadata         m_metadata;
        std::filesystem::path m_path;

    public:
        Asset(AssetMetadata metadata, std::filesystem::path path)
            : m_metadata(metadata), m_path(path) {}

        const String& getUuid() const { return m_metadata.m_uuid; }
        const String& GetName() const { return m_metadata.m_name; }
        const String& getFileId() const { return m_metadata.m_fileId; }
        virtual void  _polyHolder() {}
    };

    class IFRIT_APIDECL AssetImporter
    {
    protected:
        AssetManager* m_assetManager = nullptr;

    public:
        AssetImporter(AssetManager* manager)
            : m_assetManager(manager) {}
        virtual AssetMetadata requestDependencyMeta(const std::filesystem::path& path) { return AssetMetadata(); }
        virtual void          ImportAsset(const std::filesystem::path& path, AssetMetadata& metadata) = 0;
        virtual void          ProcessMetadata(AssetMetadata& metadata)                                = 0;
        virtual Vec<String>   GetSupportedExtensionNames()                                            = 0;
    };

    class IFRIT_APIDECL AssetManager
    {
    private:
        HashMap<String, Ref<Asset>>         m_assets;
        HashMap<String, String>             m_nameToUuid;
        HashMap<String, Ref<AssetImporter>> m_importers;
        HashMap<String, String>             m_extensionImporterMap;
        std::filesystem::path               basePath;
        IApplication*                       m_app;

    private:
        String MetadataSerialization(AssetMetadata& metadata);
        void   MetadataDeserialization(const String& serialized, AssetMetadata& metadata);

    public:
        AssetManager(std::filesystem::path path, IApplication* app);
        void                 LoadAsset(const std::filesystem::path& path);
        void                 LoadAssetDirectory(const std::filesystem::path& path);
        inline void          LoadAssetDirectory() { LoadAssetDirectory(basePath); }
        inline IApplication* GetApplication() { return m_app; }
        void                 RegisterImporter(const String& extensionName, Ref<AssetImporter> importer);
        void                 RegisterAsset(Ref<Asset> asset);
        Ref<Asset>           RequestAssetIntenal(const std::filesystem::path& path);

        template <typename T>
        Ref<T> requestAsset(const std::filesystem::path& path)
        {
            auto asset = RequestAssetIntenal(path);
            if (!asset)
            {
                return nullptr;
            }
            return std::dynamic_pointer_cast<T>(asset);
        }

        template <typename T>
        Ref<T> GetAsset(const String& uuid)
        {
            auto it = m_assets.find(uuid);
            if (it == m_assets.end())
            {
                return nullptr;
            }
            return std::dynamic_pointer_cast<T>(it->second);
        }

        template <typename T>
        Ref<T> GetAssetByName(const String& name)
        {
            auto it = m_nameToUuid.find(name);
            if (it == m_nameToUuid.end())
            {
                return nullptr;
            }
            return GetAsset<T>(it->second);
        }
    };
} // namespace Ifrit::Core
