#include "ifrit/geomproc/pointcloud/PointCloudTransforms.h"
using namespace Ifrit::Math;

namespace Ifrit::GeometryProc::PointCloud
{
    IFRIT_APIDECL void MoveCenterTo(PointCloudDescriptor& pcDesc, const Vector3f& newCenter)
    {
        if (pcDesc.m_Count == 0 || pcDesc.m_Points == nullptr)
            return;

        Vector3f currentCenter(0.0f, 0.0f, 0.0f);
        for (u32 i = 0; i < pcDesc.m_Count; ++i)
        {
            currentCenter += pcDesc.m_Points[i];
        }
        currentCenter /= static_cast<f32>(pcDesc.m_Count);

        Vector3f offset = newCenter - currentCenter;
        for (u32 i = 0; i < pcDesc.m_Count; ++i)
        {
            pcDesc.m_Points[i] += offset;
        }
    }

    IFRIT_APIDECL void NormalizeToLongestAxisAABB(
        PointCloudDescriptor& pcDesc, const Vector3f& minBound, const Vector3f& maxBound)
    {
        if (pcDesc.m_Count == 0 || pcDesc.m_Points == nullptr)
            return;

        Vector3f aabbSize    = maxBound - minBound;
        f32      longestAxis = std::max({ aabbSize.x, aabbSize.y, aabbSize.z });

        if (longestAxis <= 0.0f)
            return;

        for (u32 i = 0; i < pcDesc.m_Count; ++i)
        {
            pcDesc.m_Points[i] = (pcDesc.m_Points[i] - minBound) / longestAxis;
        }
    }

} // namespace Ifrit::GeometryProc::PointCloud