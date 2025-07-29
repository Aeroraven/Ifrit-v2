
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

#include "ifrit/runtime/base/Camera.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/core/math/linalg/LinalgOps.h"

using namespace Ifrit::Math;

namespace Ifrit::Runtime
{
    IFRIT_APIDECL Matrix4x4f Camera::GetWorldToCameraMatrix() const
    {
        auto     p              = GetParent();
        auto     transform      = p->GetComponent<Transform>();
        auto     pos            = transform->GetPosition();
        auto     rot            = transform->GetRotation();
        Vector4f frontRaw       = Vector4f{ 0.0f, 0.0f, 1.0f, 0.0f };
        auto     rotationMatrix = EulerAngleToMatrix(rot);
        auto     front          = MatMul(rotationMatrix, frontRaw);
        auto     upRaw          = Vector4f{ 0.0f, 1.0f, 0.0f, 0.0f };
        auto     up             = MatMul(rotationMatrix, upRaw);
        auto     center         = pos + Vector3f{ front.x, front.y, front.z };
        return (LookAt(Vector3f{ pos.x, pos.y, pos.z }, center, Vector3f{ up.x, up.y, up.z }));
    }
    IFRIT_APIDECL Matrix4x4f Camera::GetProjectionMatrix() const
    {
        if (mType == CameraType::Perspective)
        {
            return (PerspectiveNegateY(mFov, mAspect, mNear, mFar));
        }
        else
        {
            return (OrthographicNegateY(mOrthoSpaceSize, mAspect, mNear, mFar));
        }
    }

    IFRIT_APIDECL Vector4f Camera::GetFront() const
    {
        auto     p              = GetParent();
        auto     transform      = p->GetComponent<Transform>();
        auto     pos            = transform->GetPosition();
        auto     rot            = transform->GetRotation();
        Vector4f frontRaw       = Vector4f{ 0.0f, 0.0f, 1.0f, 0.0f };
        auto     rotationMatrix = EulerAngleToMatrix(rot);
        auto     front          = MatMul(rotationMatrix, frontRaw);
        return front;
    }

} // namespace Ifrit::Runtime