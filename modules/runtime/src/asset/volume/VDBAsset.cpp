#include "ifrit/runtime/asset/volume/VDBAsset.h"
#include "ifrit.internal/runtime/asset/volume/VDBAssetInternal.h"
#include "ifrit/geomproc/pointcloud/PointCloudTransforms.h"
namespace Ifrit::Runtime
{
    VDBAsset::VDBAsset(VDBAssetInternalData* internalData) : VolumeAsset(AssetMetadata()), mInternalData(internalData)
    {
    }

    VDBAsset::~VDBAsset() { delete mInternalData; }

    Vec<Vector3f> VDBAsset::SampleAsPointCloud(const Geometry::VolumeSamplingArgs& args)
    {

        Vec<Vector3f> pointClouds =
            GeometryProc::VDB::PoissonSampleVdbZpcReference(mInternalData->mVdbData, args.mDeltaCellX, args.mPPC);
        // iDebug("Sampled {} points from VDB.", m_PointClouds.size());
        GeometryProc::PointCloud::PointCloudDescriptor pcDesc;
        pcDesc.m_Points = pointClouds.data();
        pcDesc.m_Count  = static_cast<u32>(pointClouds.size());

        if (args.mTransform_DoNormalize)
        {
            GeometryProc::PointCloud::MoveCenterTo(pcDesc, args.mTransform_MoveToCenter);
            GeometryProc::PointCloud::NormalizeToLongestAxisAABB(
                pcDesc, args.mTransform_NormMinBound, args.mTransform_NormMaxBound);
        }
        return pointClouds;
    }
} // namespace Ifrit::Runtime