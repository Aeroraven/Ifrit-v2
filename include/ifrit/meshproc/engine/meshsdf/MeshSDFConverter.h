
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/meshproc/engine/base/MeshDesc.h"
#include <cstdint>
#include <vector>

#ifndef IFRIT_MESHPROC_IMPORT
#define IFRIT_MESHPROC_API IFRIT_APIDECL
#else
#define IFRIT_MESHPROC_API IFRIT_APIDECL_IMPORT
#endif

namespace Ifrit::MeshProcLib::MeshSDFProcess {
struct SignedDistanceField {
  Vec<f32> sdfData;
  i32 width;
  i32 height;
  i32 depth;
  ifloat3 bboxMin;
  ifloat3 bboxMax;
  IFRIT_STRUCT_SERIALIZE(sdfData, width, height, depth, bboxMin, bboxMax);
};

IFRIT_MESHPROC_API void convertMeshToSDF(const MeshDescriptor &meshDesc, SignedDistanceField &sdf, u32 sdfWidth,
                                         u32 sdfHeight, u32 sdfDepth);

} // namespace Ifrit::MeshProcLib::MeshSDFProcess