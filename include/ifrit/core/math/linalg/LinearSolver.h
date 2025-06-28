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
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/core/base/IfritBase.h"
#include <cmath>
#include <numbers>

namespace Ifrit::Math::LinAlg
{
    template <u32 R, u32 C>
    IF_FORCEINLINE Array<f32, C> JacobiSolver(
        const Matrixg<f32, R, C>& a, const Array<f32, R>& b, const Array<f32, C>& x0, u32 maxIters, f32 errorTolerance)
    {
        Array<f32, C> x = x0;
        for (u32 iter = 0; iter < maxIters; ++iter)
        {
            Array<f32, C> xNew = x;

            for (u32 i = 0; i < R; ++i)
            {
                f32 sum = 0.0f;
                for (u32 j = 0; j < C; ++j)
                {
                    if (i != j)
                    {
                        sum += a[i][j] * x[j];
                    }
                }
                xNew[i] = (b[i] - sum) / a[i][i];
            }

            if (VectorOps::Norm(xNew - x) < errorTolerance)
            {
                return xNew;
            }

            x = xNew;
        }
        return x;
    }

    template <u32 R, u32 C>
    IF_FORCEINLINE Array<f32, C> GaussSeidelSolver(
        const Matrixg<f32, R, C>& a, const Array<f32, R>& b, const Array<f32, C>& x0, u32 maxIters, f32 errorTolerance)
    {
        Array<f32, C> x = x0;
        for (u32 iter = 0; iter < maxIters; ++iter)
        {
            Array<f32, C> xNew = x;

            for (u32 i = 0; i < R; ++i)
            {
                f32 sum = 0.0f;
                for (u32 j = 0; j < C; ++j)
                {
                    if (i != j)
                    {
                        sum += a[i][j] * xNew[j];
                    }
                }
                xNew[i] = (b[i] - sum) / a[i][i];
            }

            if (VectorOps::Norm(xNew - x) < errorTolerance)
            {
                return xNew;
            }

            x = xNew;
        }
        return x;
    }
} // namespace Ifrit::Math::LinAlg