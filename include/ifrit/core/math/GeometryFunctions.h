
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
#include "../platform/ApiConv.h"
#include "./LinalgOps.h"
#include "VectorOps.h"
#include <cmath>
#include <vector>

namespace Ifrit::Math
{

    IF_FORCEINLINE Vector4f GetFrustumBoundingSphere(f32 fovy, f32 aspect, f32 fNear, f32 fFar, Vector3f apex)
    {
        f32      halfFov        = fovy / 2.0f;
        auto     halfHeightNear = fNear * std::tan(halfFov);
        auto     halfWidthNear  = halfHeightNear * aspect;
        auto     halfHeightFar  = fFar * std::tan(halfFov);
        auto     halfWidthFar   = halfHeightFar * aspect;
        auto     radiusNear     = std::sqrt(halfHeightNear * halfHeightNear + halfWidthNear * halfWidthNear);
        auto     radiusFar      = std::sqrt(halfHeightFar * halfHeightFar + halfWidthFar * halfWidthFar);

        // let x be the Distance from the apex to the center of the sphere
        // We have ((x-zn)^2+nr^2 = r^2) and ((x-zf)^2+fr^2 = r^2)
        // So, x^2 - 2x*zn + zn^2 + nr^2 = r^2 and x^2 - 2x*zf + zf^2 + fr^2 = r^2
        // So, x^2 - 2x*zn + zn^2 + nr^2 = x^2 - 2x*zf + zf^2 + fr^2
        // So, 2x(zf - zn) = fr^2 - nr^2 + zf^2 - zn^2
        // => x = (fr^2 - nr^2 + zf^2 - zn^2) / 2(zf - zn)
        auto     l            = fFar - fNear;
        auto     x            = (radiusFar * radiusFar - radiusNear * radiusNear + l * (fFar + fNear)) / (2 * l);
        auto     zCenter      = x;
        auto     sphereRadius = std::sqrt(radiusNear * radiusNear + (x - fNear) * (x - fNear));
        Vector3f center       = { apex.x, apex.y, zCenter + apex.z };
        return { center.x, center.y, center.z, sphereRadius };
    }

    IF_FORCEINLINE void GetFrustumBoundingBoxWithRay(f32 fovy, f32 aspect, f32 zNear, f32 zFar, Matrix4x4f viewToWorld,
        Vector3f apex, Vector3f rayDir, f32 reqResultZNear, f32& resZFar, f32& resOrthoSize, Vector3f& resCenter,
        f32& resCullOrthoX, f32& resCullOrthoY)
    {
        f32           halfFov        = fovy / 2.0f;
        auto          halfHeightNear = zNear * std::tan(halfFov);
        auto          halfWidthNear  = halfHeightNear * aspect;
        auto          halfHeightFar  = zFar * std::tan(halfFov);
        auto          halfWidthFar   = halfHeightFar * aspect;

        Vec<Vector4f> worldSpacePts;
        worldSpacePts.push_back({ halfWidthNear, halfHeightNear, zNear, 1.0f });
        worldSpacePts.push_back({ -halfWidthNear, halfHeightNear, zNear, 1.0f });
        worldSpacePts.push_back({ halfWidthNear, -halfHeightNear, zNear, 1.0f });
        worldSpacePts.push_back({ -halfWidthNear, -halfHeightNear, zNear, 1.0f });
        worldSpacePts.push_back({ halfWidthFar, halfHeightFar, zFar, 1.0f });
        worldSpacePts.push_back({ -halfWidthFar, halfHeightFar, zFar, 1.0f });
        worldSpacePts.push_back({ halfWidthFar, -halfHeightFar, zFar, 1.0f });
        worldSpacePts.push_back({ -halfWidthFar, -halfHeightFar, zFar, 1.0f });

        // the change of ortho size causes csm flickering.
        // Using diagonal of the frustum bounding box instead of ortho size
        // Ref: https://zhuanlan.zhihu.com/p/116731971

        auto farDist  = Distance(worldSpacePts[4], worldSpacePts[3]);
        auto diagDist = Distance(worldSpacePts[4], worldSpacePts[7]);
        auto maxDist  = std::max(farDist, diagDist);

        for (auto& pt : worldSpacePts)
        {
            auto bpt = MatMul(viewToWorld, pt);
            pt       = bpt;
        }

        f32           projMinX = std::numeric_limits<f32>::max();
        f32           projMaxX = -std::numeric_limits<f32>::max();
        f32           projMinY = std::numeric_limits<f32>::max();
        f32           projMaxY = -std::numeric_limits<f32>::max();
        f32           projMinZ = std::numeric_limits<f32>::max();
        f32           projMaxZ = -std::numeric_limits<f32>::max();

        Vector3f      dLookAtCenter = Vector3f{ 0.0f, 0.0f, 0.0f };
        Vector3f      dUp           = Vector3f{ 0.0f, 1.0f, 0.0f };
        Vector3f      dRay          = rayDir;

        Matrix4x4f    dTestView        = LookAt(dLookAtCenter, dRay, dUp);
        Matrix4x4f    dTestViewToWorld = Inverse4(dTestView);

        Vec<Vector4f> viewSpacePts;
        for (auto& pt : worldSpacePts)
        {
            auto viewSpacePt = MatMul(dTestView, pt);
            viewSpacePts.push_back(viewSpacePt);
            projMinX = std::min(projMinX, viewSpacePt.x);
            projMaxX = std::max(projMaxX, viewSpacePt.x);
            projMinY = std::min(projMinY, viewSpacePt.y);
            projMaxY = std::max(projMaxY, viewSpacePt.y);
            projMinZ = std::min(projMinZ, viewSpacePt.z);
            projMaxZ = std::max(projMaxZ, viewSpacePt.z);
        }

        // AABB center in viewspace
        auto center =
            Vector3f{ (projMinX + projMaxX) / 2.0f, (projMinY + projMaxY) / 2.0f, (projMinZ + projMaxZ) / 2.0f };
        auto orthoSize     = std::max(projMaxX - projMinX, projMaxY - projMinY);
        auto orthoSizeZ    = (projMaxZ - projMinZ) * 0.5f + reqResultZNear;
        auto orthoSizeZFar = (projMaxZ - projMinZ) + reqResultZNear;
        auto worldCenter   = MatMul(dTestViewToWorld, Vector4f{ center.x, center.y, center.z, 1.0f });
        auto reqCamPos     = Vector3f{ worldCenter.x - orthoSizeZ * dRay.x, worldCenter.y - orthoSizeZ * dRay.y,
            worldCenter.z - orthoSizeZ * dRay.z };
        // Return
        resZFar       = orthoSizeZFar;
        resOrthoSize  = maxDist;
        resCullOrthoX = projMaxX - projMinX;
        resCullOrthoY = projMaxY - projMinY;
        resCenter     = { reqCamPos.x, reqCamPos.y, reqCamPos.z };
    }
} // namespace Ifrit::Math