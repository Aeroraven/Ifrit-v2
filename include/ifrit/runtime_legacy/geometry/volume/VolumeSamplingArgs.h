#pragma once

#include "ifrit/runtime/common/Pch.h"

namespace Ifrit::Runtime::Geometry
{
    struct IFRIT_APIDECL VolumeSamplingArgs
    {
        f32      mDeltaCellX;
        u32      mPPC;
        bool     mTransform_DoNormalize;
        Vector3f mTransform_MoveToCenter;
        Vector3f mTransform_NormMinBound;
        Vector3f mTransform_NormMaxBound;
    };
} // namespace Ifrit::Runtime::Geometry