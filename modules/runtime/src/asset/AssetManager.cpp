#include "ifrit/runtime/asset/Asset.h"

namespace Ifrit::Runtime
{
    AssetMetadata AssetManager::AllocateMetadata(const String& name)
    {
        AssetMetadata metadata;
        metadata.mName = name;
        metadata.mGuid = GUID::Generate();
        return metadata;
    }
} // namespace Ifrit::Runtime