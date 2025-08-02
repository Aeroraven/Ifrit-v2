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
                metadataList.push_back(asset->mMetadata);
            }
        }
        return metadataList;
    }

    IFRIT_APIDECL EAssetRegistrationResult AssetManager::TryRegisterAsset(Owner<Asset> asset)
    {
        if (!asset)
            return EAssetRegistrationResult::InvalidArgument;

        auto& metadata = asset->mMetadata;
        if (mNameToIndex.contains(metadata.mName) || mGuidToIndex.contains(metadata.mGuid))
        {
            if (typeid(*asset) != typeid(*mAssets[mNameToIndex[metadata.mName]]))
            {
                IF_LOG_ERROR("AssetManager", "Asset with name {} or GUID {} already exists with different type.",
                    metadata.mName, metadata.mGuid.ToString());
                return EAssetRegistrationResult::Conflict;
            }
            IF_LOG_WARNING(
                "AssetManager", "Asset with name {} or GUID {} already exists.", metadata.mName, metadata.mGuid.ToString());
            return EAssetRegistrationResult::AlreadyRegistered;
        }

        mNameToIndex[metadata.mName] = SizeCast<u32>(mAssets.size());
        mGuidToIndex[metadata.mGuid] = SizeCast<u32>(mAssets.size());
        mAssets.push_back(std::move(asset));
        IF_LOG_INFO("AssetManager", "Registered asset {} with GUID {}.", metadata.mName, metadata.mGuid.ToString());
        return EAssetRegistrationResult::Success;
    }
} // namespace Ifrit::Runtime
