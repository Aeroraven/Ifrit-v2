#include "ifrit/runtime/asset/Asset.h"

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
} // namespace Ifrit::Runtime
