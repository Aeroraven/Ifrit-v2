#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/runtime/base/Material.h"

namespace Ifrit::Runtime
{

    class IFRIT_APIDECL IF_CLASS() MaterialAsset : public Asset
    {
    public:
        using Asset::Asset;

        virtual Material*         GetMaterial() = 0;
        inline virtual EAssetType GetAsseType() const final { return EAssetType::Material; }
    };
} // namespace Ifrit::Runtime
