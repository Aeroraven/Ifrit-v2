#pragma once
#include "ifrit/runtime/asset/Asset.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() VDBAssetImporter : public IAssetImporter
    {
    public:
        virtual Owner<Asset> ImportAsset(const String& relativePath) override;
    };

} // namespace Ifrit::Runtime
