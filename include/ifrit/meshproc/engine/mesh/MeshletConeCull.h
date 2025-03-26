
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "MeshClusterBase.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/ApiConv.h"
#include <meshoptimizer/src/meshoptimizer.h>
#include <vector>

#ifndef IFRIT_MESHPROC_IMPORT
    #define IFRIT_MESHPROC_API IFRIT_APIDECL
#else
    #define IFRIT_MESHPROC_API IFRIT_APIDECL_IMPORT
#endif

namespace Ifrit::MeshProcLib::MeshProcess
{

    class IFRIT_MESHPROC_API MeshletConeCullProc
    {

    public:
        void CreateNormalCones(const MeshDescriptor& meshDesc, const Vec<Vector4i>& meshlets,
            const Vec<u32>& meshletVertices, const Vec<uint8_t>& meshletTriangles,
            Vec<Vector4f>& normalConeAxisCutoff, Vec<Vector4f>& normalConeApex,
            Vec<Vector4f>& boundSphere);
    };

} // namespace Ifrit::MeshProcLib::MeshProcess