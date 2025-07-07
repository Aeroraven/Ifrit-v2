#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/geomproc/base/MeshProcBase.h"
#include "ifrit/core/math/VectorGenerics.h"
#include <any>

namespace Ifrit::GeometryProc::PointCloud
{

    struct PointCloudDescriptor
    {
        Vector3f* m_Points = nullptr;
        u32       m_Count  = 0;
    };

} // namespace Ifrit::GeometryProc::PointCloud