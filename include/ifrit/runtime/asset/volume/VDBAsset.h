#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/asset/VolumeAsset.h"

namespace Ifrit::Runtime
{

    struct VDBAssetInternalData;
    class IFRIT_APIDECL IF_CLASS() VDBAsset : public VolumeAsset
    {
    private:
        VDBAssetInternalData* mInternalData = nullptr;

    public:
        VDBAsset(VDBAssetInternalData* internalData);
        ~VDBAsset() override;
        virtual Vec<Vector3f> SampleAsPointCloud(const Geometry::VolumeSamplingArgs& args) override;
    };
} // namespace Ifrit::Runtime
