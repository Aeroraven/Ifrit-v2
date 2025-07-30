
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
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/core/algo/Guid.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/AssetReference.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/typing/Traits.h"
#include <filesystem>
namespace Ifrit::Runtime
{
    IF_CONSTEXPR const char* cMetadataFileExtension = ".meta";

    using AssetPath = std::filesystem::path;
    class AssetManager;
    class AssetImporter;

    class IFRIT_APIDECL AssetImporter
    {
    protected:
        AssetManager* mAssetManager = nullptr;

    public:
        AssetImporter(AssetManager* manager) : mAssetManager(manager) {}
        virtual void        ImportAsset(const std::filesystem::path& path, AssetMetadata& metadata) = 0;
        virtual void        ProcessMetadata(AssetMetadata& metadata)                                = 0;
        virtual Vec<String> GetSupportedExtensionNames()                                            = 0;
    };

    class IFRIT_APIDECL IF_CLASS() AssetManager : public NonCopyable
    {
    public:
        IF_PROPERTY()
        Vec<Owner<Asset>> mAssets;

    private:
        // For faster lookup
        HashMap<String, u32> mNameToIndex;
        HashMap<GUID, u32>   mGuidToIndex;

    private:
        HashMap<String, String> mExtensionImporterMap;
        std::filesystem::path   mBasePath;
        IApplication*           mApp;

    private:
        AssetMetadata AllocateMetadata(const String& name);

    public:
        AssetManager(std::filesystem::path path, IApplication* app) : mBasePath(path), mApp(app) {}
        inline IApplication* GetApplication() { return mApp; }

        template <typename T, typename... Args>
            requires(std::is_base_of<Asset, T>::value && IConceptIsConstructible<T, Args...>)
        T* CreateAsset(const String& name, Args&&... args)
        {
            Owner<T> asset                       = MakeOwner<T>(std::forward<Args>(args)...);
            asset->mMetadata                     = AllocateMetadata(name);
            mNameToIndex[asset->mMetadata.mName] = SizeCast<u32>(mAssets.size());
            mGuidToIndex[asset->mMetadata.mGuid] = SizeCast<u32>(mAssets.size());
            auto ptr                             = asset.get();
            mAssets.push_back(std::move(asset));
            return ptr;
        }

        template <typename T> T* GetAsset(const GUID& uuid)
        {
            auto it = mGuidToIndex.find(uuid);
            if (it == mGuidToIndex.end())
            {
                return nullptr;
            }
            return ForcedCheckedCast<T>(mAssets[it->second].get());
        }

        template <typename T> T* GetAssetByName(const String& name)
        {
            auto it = mNameToIndex.find(name);
            if (it == mNameToIndex.end())
            {
                return nullptr;
            }
            return ForcedCheckedCast<T>(mAssets[it->second].get());
        }
    };
} // namespace Ifrit::Runtime
