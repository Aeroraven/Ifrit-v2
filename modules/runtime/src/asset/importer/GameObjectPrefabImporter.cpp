#include "ifrit/runtime/asset/importer/GameObjectPrefabImporter.h"
#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/asset/Asset.h"
#include <filesystem>
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Runtime
{

    Owner<Asset> GameObjectPrefabImporter::ImportAsset(const String& relativePath)
    {
        // Read the content of the prefab file
        auto assetPath = GetActiveApplication()->GetAssetRegistry()->GetAbsPath(relativePath);
        if (!std::filesystem::exists(assetPath))
        {
            IF_LOG_CRITICAL("GameObjectPrefabImporter", "Prefab file does not exist: {}", assetPath);
            return nullptr;
        }
        String content = ReadTextFile(assetPath);
        if (content.empty())
        {
            IF_LOG_CRITICAL("GameObjectPrefabImporter", "Prefab file is empty: {}", assetPath);
            return nullptr;
        }
        return MakeOwner<GameObjectPrefabAsset>(content);
    }

} // namespace Ifrit::Runtime