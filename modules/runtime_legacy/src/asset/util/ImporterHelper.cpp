#include "ifrit/runtime/asset/util/ImporterHelper.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/asset/importer/VDBAssetImpoter.h"
namespace Ifrit::Runtime
{
    IFRIT_RUNTIME_API void RegisterCommonImporters()
    {
        auto assetRegistry = GetActiveApplication()->GetAssetRegistry();
        assetRegistry->RegisterImporter("VDBImporter", MakeOwner<VDBAssetImporter>());
    }
} // namespace Ifrit::Runtime