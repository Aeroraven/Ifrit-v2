#include "ifrit/runtime/physics/artemis/mpm/visualization/ProceduralMPMMeshAsset.h"
#include "ifrit/runtime/physics/artemis/mpm/visualization/ProceduralMPMMesh.h"
#include "ifrit/runtime/base/ApplicationInterface.h"

namespace Ifrit::Runtime::Artemis
{
    struct ProceduralMPMMeshAssetInternal
    {
        Owner<ProceduralMPMMesh> mMesh;
    };

    IFRIT_APIDECL ProceduralMPMMeshAsset::ProceduralMPMMeshAsset(
        u32 maxParticles, u32 maxIndices, Vector4i gridSize, Vector3f minBound, Vector3f maxBound)
        : mData(new ProceduralMPMMeshAssetInternal())
    {
        auto rhi     = GetActiveApplication()->GetRhi();
        mData->mMesh = MakeOwner<ProceduralMPMMesh>();
        mData->mMesh->Init(rhi, maxParticles, maxIndices, gridSize, minBound, maxBound);
    }

    IFRIT_APIDECL       ProceduralMPMMeshAsset::~ProceduralMPMMeshAsset() { delete mData; }

    IFRIT_APIDECL Mesh* ProceduralMPMMeshAsset::GetMesh() { return mData->mMesh.get(); }
} // namespace Ifrit::Runtime::Artemis