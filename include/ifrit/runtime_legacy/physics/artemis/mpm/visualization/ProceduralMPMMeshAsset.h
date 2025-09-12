#pragma once
#include "ifrit/runtime/asset/MeshAsset.h"

namespace Ifrit::Runtime::Artemis
{
    struct ProceduralMPMMeshAssetInternal;
    class IFRIT_RUNTIME_API IF_CLASS() ProceduralMPMMeshAsset : public MeshAsset
    {
    public:
        ProceduralMPMMeshAsset(
            u32 maxParticles, u32 maxIndices, Vector4i gridSize, Vector3f minBound, Vector3f maxBound);
        ~ProceduralMPMMeshAsset();
        virtual Mesh* GetMesh() override;

    private:
        ProceduralMPMMeshAssetInternal* mData;
    };

} // namespace Ifrit::Runtime::Artemis