
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
#include "ifrit/core/reflection/PropertyUIControl.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/algo/Guid.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include <memory>
#include <string>

namespace Ifrit::Runtime
{
    enum class EAssetReferencingType : u8
    {
        Unknown,
        Internal,
        Imported
    };

    enum class EAssetType : u8
    {
        General,
        Texture,
        Material,
        Mesh,
        Prefab,
        Shader
    };

    struct IF_CLASS() AssetReferenceId
    {
        IF_PROPERTY()
        EAssetReferencingType mType;

        IF_PROPERTY()
        GUID mGuid;

        IF_PROPERTY()
        String mRelativePath;

        bool   operator==(const AssetReferenceId& other) const
        {
            if (mType != other.mType)
                return false;
            if (mGuid != other.mGuid)
            {
                if (mType != EAssetReferencingType::Internal)
                    return mRelativePath == other.mRelativePath;
                return false;
            }
            else
            {
                return true;
            }
        }

        IFRIT_APIDECL void GetUIEditingHandle(const Reflection::PropertyMetadata& propMeta);
        IFRIT_APIDECL void DoSerialize(Reflection::Archive* archive) const;
        IFRIT_APIDECL void DoDeserialize(Reflection::Archive* archive);
    };

    struct IF_CLASS() AssetMetadata
    {

        IF_PROPERTY()
        GUID mGuid;

        IF_PROPERTY()
        String mName;

        IF_PROPERTY()
        String mExternalPath;

        IF_PROPERTY()
        String mImporter = "";

        IF_PROPERTY()
        EAssetReferencingType mReferencingType = EAssetReferencingType::Unknown;

        EAssetType            mAssetType = EAssetType::General;
    };

    class IFRIT_APIDECL IF_CLASS() Asset
    {
    public:
        IF_PROPERTY()
        AssetMetadata mMetadata;

    public:
        Asset()          = default;
        virtual ~Asset() = default;

        Asset(AssetMetadata metadata) : mMetadata(metadata) {}
        const GUID&               GetGuid() const { return mMetadata.mGuid; }
        const String&             GetName() const { return mMetadata.mName; }
        const String&             GetExternalPath() const { return mMetadata.mExternalPath; }
        virtual void              _PolyHolder() {}

        const AssetReferenceId    GetAssetReference() const;
        inline virtual EAssetType GetAsseType() const { return EAssetType::General; }
    };

    class IF_CLASS() InternalAssetHolder
    {
    public:
        IF_PROPERTY()
        Owner<Asset> mAsset;
    };

} // namespace Ifrit::Runtime