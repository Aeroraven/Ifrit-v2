#include "ifrit/runtime/base/MeshComponent.h"
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/core/logging/Logging.h"

namespace Ifrit::Runtime
{

    IFRIT_APIDECL void  MeshFilter::SetMeshSource(MeshAsset* p) { mMesh = p->GetAssetReference(); }
    IFRIT_APIDECL Mesh* MeshFilter::GetMesh()
    {

        auto app           = GetActiveApplication();
        auto assetRegistry = app->GetAssetRegistry();
        auto meshAsset     = assetRegistry->GetAsset<MeshAsset>(mMesh.mGuid);
        if (meshAsset == nullptr)
            return nullptr;
        return meshAsset->GetMesh();
    }

    IFRIT_APIDECL Material* MeshRenderer::GetMaterial()
    {
        auto app           = GetActiveApplication();
        auto assetRegistry = app->GetAssetRegistry();
        auto materialAsset = assetRegistry->GetAsset<MaterialAsset>(mMaterial.mGuid);
        if (materialAsset == nullptr)
            return nullptr;
        return materialAsset->GetMaterial();
    }
    IFRIT_APIDECL void MeshRenderer::SetMaterialSource(MaterialAsset* p) { mMaterial = p->GetAssetReference(); }

} // namespace Ifrit::Runtime
