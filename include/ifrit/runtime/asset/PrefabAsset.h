#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/runtime/base/Scene.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() PrefabAsset : public Asset
    {
    public:
        using Asset::Asset;

        virtual void              InstantiatePrefab(SceneNode* node) = 0;
        inline virtual EAssetType GetAsseType() const final { return EAssetType::Prefab; }
    };

} // namespace Ifrit::Runtime
