#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/geomproc/base/MeshProcBase.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/geomproc/vdb/VdbBase.h"

namespace Ifrit::GeometryProc::VDB
{

    IFRIT_GEOMPROC_API Vec<Vector3f> PoissonSampleVdbZpcReference(const VDBDescriptor& vdbDesc, f32 dx, u32 ppc);

} // namespace Ifrit::GeometryProc::VDB