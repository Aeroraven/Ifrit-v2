#pragma once
#include "ifrit/geomproc/vdb/VdbSampler.h"

namespace Ifrit::Runtime
{
    struct VDBAssetInternalData
    {
        GeometryProc::VDB::VDBDescriptor mVdbData;
    };
} // namespace Ifrit::Runtime