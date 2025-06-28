
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
#include "ifrit/core/math/linalg/LinalgOps.h"
#include "ifrit/core/serialization/MathTypeSerialization.h"
#include "ifrit/core/serialization/SerialInterface.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/meshproc/engine/base/MeshDesc.h"
#include "ifrit/meshproc/engine/base/MeshProcBase.h"

namespace Ifrit::MeshProcLib::MeshSDFProcess
{
    enum class SDFGenerateMethod
    {
        Trivial,
        RayTracing
    };

    struct SignedDistanceField
    {
        Vec<f32> sdfData;
        i32      width;
        i32      height;
        i32      depth;
        Vector3f bboxMin;
        Vector3f bboxMax;
        IFRIT_STRUCT_SERIALIZE(sdfData, width, height, depth, bboxMin, bboxMax);
    };

    struct CompactSignedDistanceField
    {
        Vec<u8>  sdfData;
        i32      width;
        i32      height;
        i32      depth;
        Vector3f bboxMin;
        Vector3f bboxMax;
        f32      m_SdfMin;
        f32      m_SdfMax;
        IFRIT_STRUCT_SERIALIZE(sdfData, width, height, depth, bboxMin, bboxMax, m_SdfMin, m_SdfMax);
    };

    struct CompactSignedDistanceFieldMeta
    {
        i32      width;
        i32      height;
        i32      depth;
        Vector3f bboxMin;
        Vector3f bboxMax;
        f32      m_SdfMin;
        f32      m_SdfMax;
        IFRIT_STRUCT_SERIALIZE(width, height, depth, bboxMin, bboxMax, m_SdfMin, m_SdfMax);
    };

    IFRIT_MESHPROC_API void ConvertMeshToSDF(const MeshDescriptor& meshDesc, SignedDistanceField& sdf, u32 sdfWidth,
        u32 sdfHeight, u32 sdfDepth, SDFGenerateMethod method, bool twoSided);

    IFRIT_MESHPROC_API void CompactSDF(const SignedDistanceField& sdf, CompactSignedDistanceField& compactSdf);

} // namespace Ifrit::MeshProcLib::MeshSDFProcess