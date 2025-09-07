#include "ifrit/runtime/asset/volume/VDBAsset.h"
#include "ifrit/runtime/asset/importer/VDBAssetImpoter.h"
#include "ifrit.internal/runtime/asset/volume/VDBAssetInternal.h"
#include "ifrit/core/file/FileOps.h"

namespace Ifrit::Runtime
{
    Owner<Asset> VDBAssetImporter::ImportAsset(const String& relativePath)
    {
        auto vdbInternalData      = new VDBAssetInternalData();
        auto vdbFileData          = ReadBinaryFile(relativePath);
        vdbInternalData->mVdbData = GeometryProc::VDB::LoadVdbFromString(vdbFileData);
        auto vdbAsset             = MakeOwner<VDBAsset>(vdbInternalData);
        return vdbAsset;
    }
} // namespace Ifrit::Runtime