#pragma once
#include "ifrit/runtime/asset/PrefabAsset.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() GameObjectPrefabAsset : public PrefabAsset
    {
    public:
        using PrefabAsset::PrefabAsset;

        virtual void InstantiatePrefab(SceneNode* node) override {}
    };

} // namespace Ifrit::Runtime
