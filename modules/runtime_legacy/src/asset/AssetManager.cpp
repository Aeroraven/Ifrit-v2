#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Runtime
{
    AssetMetadata AssetManager::AllocateMetadata(const String& name)
    {
        AssetMetadata metadata;
        metadata.mName = name;
        metadata.mGuid = GUID::Generate();
        return metadata;
    }

    IFRIT_APIDECL String AssetManager::GetAbsPath(const String& relativePath) const
    {
        if (relativePath.empty())
        {
            return mBasePath.string();
        }
        std::filesystem::path absPath = mBasePath / relativePath;
        return absPath.string();
    }

    IFRIT_APIDECL void AssetManager::ImportAssetImpl(
        const String& relativePath, const String& importerId, const String& newName, const GUID& newGuid)
    {
        // this function call does not check safety guards
        auto fullPath                     = GetAbsPath(relativePath);
        auto importer                     = mImporters[importerId].get();
        auto asset                        = importer->ImportAsset(fullPath);
        asset->mMetadata.mName            = newName;
        asset->mMetadata.mGuid            = newGuid;
        asset->mMetadata.mExternalPath    = relativePath;
        asset->mMetadata.mReferencingType = EAssetReferencingType::Imported;
        asset->mMetadata.mImporter        = importerId;
        mAssets.push_back(std::move(asset));
        mGuidToIndex[newGuid] = SizeCast<u32>(mAssets.size() - 1);
        mNameToIndex[newName] = SizeCast<u32>(mAssets.size() - 1);
    }

    IFRIT_APIDECL Vec<AssetMetadata> AssetManager::GetAllAssetMetadata() const
    {
        Vec<AssetMetadata> metadataList;
        metadataList.reserve(mAssets.size());
        for (const auto& asset : mAssets)
        {
            if (asset)
            {
                auto metaCopy       = asset->mMetadata;
                metaCopy.mAssetType = asset->GetAsseType();
                metadataList.push_back(metaCopy);
            }
        }
        return metadataList;
    }

    IFRIT_APIDECL AssetRegistrationResult AssetManager::TryImportAssetWithRenaming(
        const String& relativePath, const String& importerId, const String& newName, const GUID& newGuid)
    {
        AssetRegistrationResult ret;
        auto                    absPath = GetAbsPath(relativePath);
        if (!std::filesystem::exists(absPath))
        {
            IF_LOG_CRITICAL("AssetManager", "Asset file does not exist: {}", absPath);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }
        auto importerIt = mImporters.find(importerId);
        if (importerIt == mImporters.end())
        {
            IF_LOG_CRITICAL("AssetManager", "Importer not found: {}", importerId);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }
        auto& importer = importerIt->second;
        if (!importer)
        {
            IF_LOG_CRITICAL("AssetManager", "Importer is null for ID: {}", importerId);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }
        // checking if the asset is already registered
        auto existingAssetName = mNameToIndex.contains(newName);
        auto existingAssetGuid = mGuidToIndex.contains(newGuid);
        if (existingAssetGuid && existingAssetName)
        {
            auto assetByName = GetAssetByName<Asset>(newName);
            auto assetByGuid = GetAsset<Asset>(newGuid);
            // emm, i don't think this is correct
            if (assetByName && assetByGuid && assetByName == assetByGuid)
            {
                IF_LOG_WARNING(
                    "AssetManager", "Asset with GUID {} already registered with name {}.", newGuid.ToString(), newName);
                ret.mCode = EAssetRegistrationResultCode::AlreadyRegistered;
                ret.mGuid = newGuid;
                ret.mName = newName;
                return ret;
            }
        }
        auto guid = newGuid;
        if (existingAssetGuid)
        {
            guid = GUID::Generate();
        }
        if (existingAssetName)
        {
            for (int dupInd = 1;; dupInd++)
            {
                auto newAssetName = newName + "_" + std::to_string(dupInd);
                if (!mNameToIndex.contains(newAssetName))
                {
                    ret.mName = newAssetName;
                    break;
                }
            }
        }
        else
        {
            ret.mName = newName;
        }
        ret.mGuid = guid;
        ret.mCode = EAssetRegistrationResultCode::Success;

        ImportAssetImpl(relativePath, importerId, ret.mName, ret.mGuid);
        return ret;
    }

    IFRIT_APIDECL AssetRegistrationResult AssetManager::TryRegisterAsset(Owner<Asset> asset)
    {
        AssetRegistrationResult ret;
        if (!asset)
        {
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            IF_LOG_CRITICAL("AssetManager", "Cannot register null asset");
            return ret;
        }

        auto& metadata  = asset->mMetadata;
        bool  nameExist = mNameToIndex.contains(metadata.mName);
        bool  guidExist = mGuidToIndex.contains(metadata.mGuid);

        if (nameExist && guidExist)
        {
            Asset* existAssetWithName = mAssets[mNameToIndex[metadata.mName]].get();
            Asset* existAssetWithGuid = mAssets[mGuidToIndex[metadata.mGuid]].get();

            if (existAssetWithName == existAssetWithGuid && typeid(*existAssetWithName) == typeid(*asset))
            {
                IF_LOG_WARNING("AssetManager", "Asset with GUID {} already registered with name {}.",
                    metadata.mGuid.ToString(), metadata.mName);
                ret.mCode = EAssetRegistrationResultCode::AlreadyRegistered;
                ret.mGuid = metadata.mGuid;
                ret.mName = metadata.mName;
                return ret;
            }
        }

        GUID   guid = metadata.mGuid;
        String name = metadata.mName;
        if (guidExist)
        {
            guid = GUID::Generate();
        }

        if (nameExist)
        {
            for (int dupInd = 1;; dupInd++)
            {
                auto newName = name + "_" + std::to_string(dupInd);
                if (!mNameToIndex.contains(newName))
                {
                    metadata.mName = newName;
                    break;
                }
            }
        }

        mNameToIndex[name] = SizeCast<u32>(mAssets.size());
        mGuidToIndex[guid] = SizeCast<u32>(mAssets.size());

        asset->mMetadata.mName = name;
        asset->mMetadata.mGuid = guid;
        mAssets.push_back(std::move(asset));
        IF_LOG_INFO("AssetManager", "Registered asset {} with GUID {}.", metadata.mName, metadata.mGuid.ToString());

        ret.mCode = EAssetRegistrationResultCode::Success;
        ret.mGuid = guid;
        ret.mName = name;
        return ret;
    }

    IFRIT_APIDECL void AssetManager::RegisterImporter(const String& importerId, Owner<IAssetImporter> importer)
    {
        if (!importer)
        {
            IF_LOG_CRITICAL("AssetManager", "Cannot register null importer for ID: {}", importerId);
            return;
        }
        if (mImporters.contains(importerId))
        {
            IF_LOG_WARNING("AssetManager", "Importer with ID {} already registered, replacing it.", importerId);
        }
        mImporters[importerId] = std::move(importer);
        IF_LOG_INFO("AssetManager", "Registered importer with ID: {}", importerId);
    }

    IFRIT_APIDECL Vec<AssetImporterPair> AssetManager::GetAllImporters() const
    {
        Vec<AssetImporterPair> importers;
        importers.reserve(mImporters.size());
        for (const auto& [id, importer] : mImporters)
        {
            if (importer)
            {
                importers.push_back({ id, importer.get() });
            }
        }
        return importers;
    }

    IFRIT_APIDECL AssetRegistrationResult AssetManager::ImportAsset(
        const String& importerId, const String& relativePath, const String& name)
    {
        AssetRegistrationResult ret{};
        auto                    guid = GUID::Generate();
        if (mNameToIndex.contains(name) || mGuidToIndex.contains(guid))
        {
            IF_LOG_CRITICAL("AssetManager", "Asset with name {} or GUID {} already exists.", name, guid.ToString());
            ret.mCode = EAssetRegistrationResultCode::Conflict;
            return ret;
        }
        auto importerIt = mImporters.find(importerId);
        if (importerIt == mImporters.end())
        {
            IF_LOG_CRITICAL("AssetManager", "Importer with ID {} not found.", importerId);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }
        auto& importer = importerIt->second;
        if (!importer)
        {
            IF_LOG_CRITICAL("AssetManager", "Importer with ID {} is null.", importerId);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }
        auto absPath = GetAbsPath(relativePath);
        if (!std::filesystem::exists(absPath))
        {
            IF_LOG_CRITICAL("AssetManager", "Asset file does not exist: {}", absPath);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }
        ImportAssetImpl(relativePath, importerId, name, guid);
        ret.mCode = EAssetRegistrationResultCode::Success;
        ret.mGuid = guid;
        ret.mName = name;
        IF_LOG_INFO("AssetManager", "Imported asset {} with GUID {} from path {}.", name, guid.ToString(), absPath);
        return ret;
    }

    AssetRegistrationResult AssetManager::ImportAssetFromAbsPath(
        const String& importerId, const String& absPath, const String& name)
    {
        AssetRegistrationResult ret{};

        // Check if the absolute path exists
        if (!std::filesystem::exists(absPath))
        {
            IF_LOG_CRITICAL("AssetManager", "Asset file does not exist: {}", absPath);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }

        // Normalize paths and check if absPath is within asset directory
        std::filesystem::path normalizedAbsPath  = std::filesystem::weakly_canonical(absPath);
        std::filesystem::path normalizedBasePath = std::filesystem::weakly_canonical(mBasePath);

        // Check if the absolute path is a child of the asset directory
        auto                  relativePath = std::filesystem::relative(normalizedAbsPath, normalizedBasePath);
        if (relativePath.empty() || relativePath.string().starts_with(".."))
        {
            IF_LOG_CRITICAL(
                "AssetManager", "Asset path {} is not within the asset directory {}", absPath, mBasePath.string());
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }

        // Check if importer exists
        auto importerIt = mImporters.find(importerId);
        if (importerIt == mImporters.end())
        {
            IF_LOG_CRITICAL("AssetManager", "Importer with ID {} not found.", importerId);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }

        auto& importer = importerIt->second;
        if (!importer)
        {
            IF_LOG_CRITICAL("AssetManager", "Importer with ID {} is null.", importerId);
            ret.mCode = EAssetRegistrationResultCode::InvalidArgument;
            return ret;
        }

        // Generate GUID and check for conflicts
        auto guid = GUID::Generate();
        if (mNameToIndex.contains(name) || mGuidToIndex.contains(guid))
        {
            IF_LOG_CRITICAL("AssetManager", "Asset with name {} or GUID {} already exists.", name, guid.ToString());
            ret.mCode = EAssetRegistrationResultCode::Conflict;
            return ret;
        }

        // Convert relative path to string and normalize separators
        String relativePathString = relativePath.string();
        std::replace(relativePathString.begin(), relativePathString.end(), '\\', '/');

        // Import the asset using the relative path
        ImportAssetImpl(relativePathString, importerId, name, guid);

        ret.mCode = EAssetRegistrationResultCode::Success;
        ret.mGuid = guid;
        ret.mName = name;

        IF_LOG_INFO(
            "AssetManager", "Imported asset {} with GUID {} from absolute path {}.", name, guid.ToString(), absPath);

        return ret;
    }
} // namespace Ifrit::Runtime
