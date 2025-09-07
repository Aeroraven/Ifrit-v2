#pragma once

#include "ifrit/core/math/linalg/LinalgOps.h"

namespace Ifrit::Math
{
    // References: https://en.wikipedia.org/wiki/List_of_moments_of_inertia

    // Planar primitives

    IF_FORCEINLINE f32 DiskMomentOfInertiaWrtCenter2D(f32 radius, f32 mass) { return 0.5f * mass * radius * radius; }

    IF_FORCEINLINE f32 RectMomentOfInertiaWrtCenter2D(f32 width, f32 height, f32 mass)
    {
        return (1.0f / 12.0f) * mass * (width * width + height * height);
    }

    // Solid primitives

    IF_FORCEINLINE Matrix3x3f SphereMomentOfInertiaWrtCenter3D(f32 radius, f32 mass)
    {
        // I = 2/5 * m * r^2
        f32        inertia       = (2.0f / 5.0f) * mass * radius * radius;
        Matrix3x3f inertiaTensor = ZeroMatrix<f32, 3, 3>();
        inertiaTensor[0][0]      = inertia;
        inertiaTensor[1][1]      = inertia;
        inertiaTensor[2][2]      = inertia;
        return inertiaTensor;
    }

    IF_FORCEINLINE Matrix3x3f CuboidMomentOfInertiaWrtCenter3D(f32 width, f32 height, f32 depth, f32 mass)
    {
        // I = 1/12 * m * (h^2 + d^2), 1/12 * m * (w^2 + d^2), 1/12 * m * (w^2 + h^2)
        Matrix3x3f inertiaTensor = ZeroMatrix<f32, 3, 3>();
        inertiaTensor[0][0]      = (1.0f / 12.0f) * mass * (height * height + depth * depth);
        inertiaTensor[1][1]      = (1.0f / 12.0f) * mass * (width * width + depth * depth);
        inertiaTensor[2][2]      = (1.0f / 12.0f) * mass * (width * width + height * height);
        return inertiaTensor;
    }

} // namespace Ifrit::Math