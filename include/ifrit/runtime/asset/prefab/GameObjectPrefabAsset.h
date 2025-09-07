#pragma once
#include "ifrit/runtime/asset/PrefabAsset.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() GameObjectPrefabAsset : public PrefabAsset
    {
    public:
        String mContent;

        GameObjectPrefabAsset() = default;
        GameObjectPrefabAsset(const String& content);
        virtual ~GameObjectPrefabAsset() override = default;

        virtual void InstantiatePrefab(SceneNode* node) override {}
    };

} // namespace Ifrit::Runtime
