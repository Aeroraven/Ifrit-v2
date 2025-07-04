#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/geomproc/base/MeshProcBase.h"
#include <any>

namespace Ifrit::GeometryProc::VDB
{
    struct VDBDescriptor
    {
        std::any m_VdbData;
    };

    IFRIT_GEOMPROC_API VDBDescriptor LoadVdbFromString(String s);

} // namespace Ifrit::GeometryProc::VDB