#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/meshproc/engine/base/MeshProcBase.h"
#include <any>

namespace Ifrit::MeshProcLib::VDB
{
    struct VDBDescriptor
    {
        std::any m_VdbData;
    };

    IFRIT_MESHPROC_API VDBDescriptor LoadVdbFromString(String s);

} // namespace Ifrit::MeshProcLib::VDB