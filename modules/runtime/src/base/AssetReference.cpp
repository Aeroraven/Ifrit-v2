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
        archive->BeginObject("__ifrit_asset_reference");
        archive->BeginObject("__ifrit_guid");
        mGuid.DoSerialize(archive);
        archive->EndObject();
        archive->BeginObject("__ifrit_path");
        archive->Serialize(mRelativePath);
        archive->EndObject();

        auto assetRegistry = GetActiveApplication()->GetAssetRegistry();
        auto asset         = assetRegistry->GetAsset<Asset>(mGuid);
        archive->BeginObject("__ifrit_asset_is_valid");
        int valid = asset ? 1 : 0;
        archive->Serialize(valid);
        archive->EndObject();

        if (valid)
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
        archive->EndObject();
    }
    IFRIT_APIDECL void AssetReferenceId::DoDeserialize(Reflection::Archive* archive)
    {
        archive->BeginObject("__ifrit_asset_reference");
        archive->BeginObject("__ifrit_guid");
        mGuid.DoSerialize(archive);
        archive->EndObject();
        archive->BeginObject("__ifrit_path");
        archive->Serialize(mRelativePath);
        archive->EndObject();

        int valid = 0;
        archive->BeginObject("__ifrit_asset_is_valid");
        archive->Serialize(valid);
        archive->EndObject();
        if (valid)
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
                if (result != EAssetRegistrationResult::Success
                    && result != EAssetRegistrationResult::AlreadyRegistered)
                {
                    IF_LOG_ERROR("AssetReferenceId", "Failed to register asset: {}", mRelativePath);
                }
            }
            else IF_UNLIKELY
            {
                IF_LOG_CRITICAL("AssetReferenceId", "Deserialized asset is null for path: {}", mRelativePath);
            }
        }
    }
} // namespace Ifrit::Runtime