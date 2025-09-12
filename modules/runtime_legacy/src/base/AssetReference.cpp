#include "ifrit/runtime/base/AssetReference.h"
#include "ifrit/runtime/base/EditorHandles.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/core/reflection/SerializeHelper.h"
namespace Ifrit::Runtime
{

    IFRIT_APIDECL void AssetReferenceId::GetUIEditingHandle(const Reflection::PropertyMetadata& propMeta)
    {
        auto& handles = GetRuntimeEditorHandles();
        if (handles.AssetReferenceHandle)
        {
            handles.AssetReferenceHandle(propMeta.Name.c_str(), *this);
        }
    }
    IFRIT_APIDECL void AssetReferenceId::DoSerialize(Reflection::Archive* archive) const
    {
        auto   assetRegistry = GetActiveApplication()->GetAssetRegistry();
        Asset* asset         = nullptr;

        auto   actualMType = mType;
        auto   actualMPath = mRelativePath;

        if (mType != EAssetReferencingType::Unknown)
        {
            asset = assetRegistry->GetAsset<Asset>(mGuid);
            if (asset != nullptr)
            {
                auto& assetMetadata = asset->mMetadata;
                actualMType         = assetMetadata.mReferencingType;
                actualMPath         = assetMetadata.mExternalPath;
            }
        }

        archive->BeginObject("__ifrit_asset_reference");
        archive->BeginObject("__ifrit_guid");
        mGuid.DoSerialize(archive);
        archive->EndObject();
        archive->BeginObject("__ifrit_path");
        archive->Serialize(actualMPath);
        archive->EndObject();

        using U     = std::underlying_type_t<EAssetReferencingType>;
        U typeValue = static_cast<U>(actualMType);
        archive->BeginObject("__ifrit_asset_referencing_type");
        archive->Serialize(typeValue);
        archive->EndObject();

        if (actualMType == EAssetReferencingType::Unknown)
        {
            IF_LOG_CRITICAL("AssetReferenceId", "AssetReferenceId is in unknown state, cannot serialize");
            archive->EndObject();
            return;
        }
        else if (actualMType == EAssetReferencingType::Empty)
        {
            archive->EndObject();
            return;
        }

        archive->BeginObject("__ifrit_asset_is_valid");
        int valid = asset ? 1 : 0;
        archive->Serialize(valid);
        archive->EndObject();

        if (valid)
        {
            if (actualMType == EAssetReferencingType::Internal)
            {
                String              serializedAsset;
                InternalAssetHolder holder;
                holder.mAsset = Owner<Asset>(asset);
                archive->BeginObject("__ifrit_asset_internal_object");
                serializedAsset = Reflection::SerializeToJSON(holder);
                archive->Serialize(serializedAsset);
                holder.mAsset.release();
                archive->EndObject();
            }
            else if (actualMType == EAssetReferencingType::Imported)
            {
                // Saving importer
                archive->BeginObject("__ifrit_asset_importer");
                archive->Serialize(asset->mMetadata.mImporter);
                archive->EndObject();
            }
        }
        archive->EndObject();
    }
    IFRIT_APIDECL void AssetReferenceId::DoDeserialize(Reflection::Archive* archive)
    {
        archive->BeginObject("__ifrit_asset_reference");
        archive->BeginObject("__ifrit_guid");
        mGuid.DoDeserialize(archive);
        archive->EndObject();
        archive->BeginObject("__ifrit_path");
        archive->Serialize(mRelativePath);
        archive->EndObject();
        using U = std::underlying_type_t<EAssetReferencingType>;
        U typeValue;
        archive->BeginObject("__ifrit_asset_referencing_type");
        archive->Serialize(typeValue);
        archive->EndObject();
        mType = static_cast<EAssetReferencingType>(typeValue);

        if (mType == EAssetReferencingType::Unknown)
        {
            IF_LOG_CRITICAL("AssetReferenceId", "AssetReferenceId is in unknown state, cannot deserialize");
            archive->EndObject();
            return;
        }
        else if (mType == EAssetReferencingType::Empty)
        {
            archive->EndObject();
            return;
        }

        int valid = 0;
        archive->BeginObject("__ifrit_asset_is_valid");
        archive->Serialize(valid);
        archive->EndObject();
        if (valid)
        {
            if (mType == EAssetReferencingType::Internal)
            {
                auto hasObject = archive->HasObject("__ifrit_asset_internal_object");
                if (!hasObject)
                {
                    IF_LOG_CRITICAL("AssetReferenceId",
                        "Missing internal asset object during deserialization, the"
                        "serialized data is corrupted");
                    return;
                }
                archive->BeginObject("__ifrit_asset_internal_object");
                String serializedAsset;
                archive->Serialize(serializedAsset);
                InternalAssetHolder holder;
                Reflection::DeserializeFromJSON(holder, serializedAsset);
                archive->EndObject();

                // Register the asset
                auto assetRegistry = GetActiveApplication()->GetAssetRegistry();
                if (holder.mAsset)
                {
                    auto result = assetRegistry->TryRegisterAsset(std::move(holder.mAsset));
                    if (result.mCode != EAssetRegistrationResultCode::Success
                        && result.mCode != EAssetRegistrationResultCode::AlreadyRegistered)
                    {
                        IF_LOG_CRITICAL("AssetReferenceId", "Failed to register asset: {}", mRelativePath);
                    }
                    mGuid = result.mGuid;
                    mType = EAssetReferencingType::Internal;
                }
                else IF_UNLIKELY
                {
                    IF_LOG_CRITICAL("AssetReferenceId", "Deserialized asset is null for path: {}", mRelativePath);
                }
            }
            else if (mType == EAssetReferencingType::Imported)
            {
                // Load importer
                String importerId;
                archive->BeginObject("__ifrit_asset_importer");
                archive->Serialize(importerId);
                archive->EndObject();

                auto assetRegistry = GetActiveApplication()->GetAssetRegistry();
                auto result =
                    assetRegistry->TryImportAssetWithRenaming(mRelativePath, importerId, mRelativePath, mGuid);
                if (result.mCode != EAssetRegistrationResultCode::Success
                    && result.mCode != EAssetRegistrationResultCode::AlreadyRegistered)
                {
                    IF_LOG_CRITICAL("AssetReferenceId", "Failed to import asset: {}", mRelativePath);
                }
                mGuid         = result.mGuid;
                mRelativePath = result.mName;
                mType         = EAssetReferencingType::Imported;
            }
            archive->EndObject();
        }
    }

    IFRIT_APIDECL const AssetReferenceId Asset::GetAssetReference() const
    {
        if (mMetadata.mReferencingType == EAssetReferencingType::Unknown)
        {
            IF_LOG_CRITICAL("Asset", "Asset metadata referencing type is unknown for asset: {}", mMetadata.mName);
            return AssetReferenceId();
        }
        AssetReferenceId ref;
        ref.mType         = mMetadata.mReferencingType;
        ref.mGuid         = mMetadata.mGuid;
        ref.mRelativePath = mMetadata.mExternalPath;
        return ref;
    }

} // namespace Ifrit::Runtime