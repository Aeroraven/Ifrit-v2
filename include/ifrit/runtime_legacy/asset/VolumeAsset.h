#pragma once
#include "ifrit/runtime/asset/Asset.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/reflection/ReflAttrs.h"
#include "ifrit/runtime/geometry/volume/VolumeSamplingArgs.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL IF_CLASS() VolumeAsset : public Asset
    {
    public:
        using Asset::Asset;

        virtual Vec<Vector3f>     SampleAsPointCloud(const Geometry::VolumeSamplingArgs& args) = 0;
        inline virtual EAssetType GetAsseType() const final { return EAssetType::VolumetricData; }
    };

} // namespace Ifrit::Runtime
