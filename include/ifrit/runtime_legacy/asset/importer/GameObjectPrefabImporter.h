#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/asset/prefab/GameObjectPrefabAsset.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() GameObjectPrefabImporter : public IAssetImporter
    {
    public:
        virtual Owner<Asset> ImportAsset(const String& relativePath) override;
    };

} // namespace Ifrit::Runtime