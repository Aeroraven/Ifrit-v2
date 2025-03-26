
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include <memory>
#include <string>

namespace Ifrit::Core
{
    struct AssetReference
    {
        String m_fileId;
        String m_uuid;
        String m_name;
        bool   m_usingAsset = false;
        IFRIT_STRUCT_SERIALIZE(m_fileId, m_uuid, m_name, m_usingAsset)

        bool operator==(const AssetReference& other) const { return m_uuid == other.m_uuid && m_name == other.m_name; }
    };

    class IFRIT_APIDECL IAssetCompatible
    {
    public:
        virtual void _PolyHolderAsset() {}
    };

    class AssetReferenceContainer
    {
    public:
        AssetReference                  m_assetReference;
        bool                            m_usingAsset = false;
        std::weak_ptr<IAssetCompatible> m_asset;

        IFRIT_STRUCT_SERIALIZE(m_assetReference, m_usingAsset)
    };

} // namespace Ifrit::Core