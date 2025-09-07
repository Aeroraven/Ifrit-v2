#pragma once
#include "ifrit/geomproc/pointcloud/PointCloudBase.h"

namespace Ifrit::GeometryProc::PointCloud
{
    IFRIT_GEOMPROC_API void MoveCenterTo(PointCloudDescriptor& pcDesc, const Vector3f& newCenter);

    IFRIT_GEOMPROC_API void NormalizeToLongestAxisAABB(
        PointCloudDescriptor& pcDesc, const Vector3f& minBound, const Vector3f& maxBound);

} // namespace Ifrit::GeometryProc::PointCloud