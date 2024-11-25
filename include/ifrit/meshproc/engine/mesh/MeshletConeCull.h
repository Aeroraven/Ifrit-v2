#pragma once
#include "MeshClusterBase.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/ApiConv.h"
#include <meshoptimizer/src/meshoptimizer.h>
#include <vector>

#ifndef IFRIT_MESHPROC_IMPORT
#define IFRIT_MESHPROC_API IFRIT_APIDECL
#else
#define IFRIT_MESHPROC_API IFRIT_APIDECL_IMPORT
#endif

namespace Ifrit::MeshProcLib::MeshProcess {

class IFRIT_MESHPROC_API MeshletConeCullProc {

public:
  void createNormalCones(const MeshDescriptor &meshDesc,
                         const std::vector<iint4> &meshlets,
                         const std::vector<uint32_t> &meshletVertices,
                         const std::vector<uint8_t> &meshletTriangles,
                         std::vector<ifloat4> &normalConeAxisCutoff,
                         std::vector<ifloat4> &normalConeApex,
                         std::vector<ifloat4> &boundSphere);
};

} // namespace Ifrit::MeshProcLib::MeshProcess