
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/LinalgOps.h"
#include "ifrit/core/serialization/MathTypeSerialization.h"
#include "ifrit/core/serialization/SerialInterface.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/meshproc/engine/base/MeshDesc.h"
#include "ifrit/meshproc/engine/base/MeshProcBase.h"

namespace Ifrit::MeshProcLib::Tetrahedralization
{

    IFRIT_MESHPROC_API FTetrahedralMeshData TetrahedralizeMesh(const MeshDescriptor& mesh);

} // namespace Ifrit::MeshProcLib::Tetrahedralization