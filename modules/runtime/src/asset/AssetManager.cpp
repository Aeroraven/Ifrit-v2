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
    IFRIT_APIDECL EAssetRegistrationResult TryRegisterAssetWithRenaming(
        Owner<Asset> asset, const String& newName, const GUID& newGuid)
    {
        return EAssetRegistrationResult::InvalidArgument;
    }

    IFRIT_APIDECL EAssetRegistrationResult AssetManager::TryRegisterAsset(Owner<Asset> asset)
    {
        if (!asset)
            return EAssetRegistrationResult::InvalidArgument;

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
                return EAssetRegistrationResult::AlreadyRegistered;
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
        return EAssetRegistrationResult::Success;
    }
} // namespace Ifrit::Runtime
