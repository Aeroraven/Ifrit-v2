
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

#include "Component.h"
#include "ifrit/core/math/VectorDefs.h"
#include "ifrit/core/serialization/MathTypeSerialization.h"
#include "ifrit/core/serialization/SerialInterface.h"
#include "ifrit/core/serialization/SerialEnumDefine.h"
#include "ifrit/core/reflection/ReflAttrs.h"

namespace Ifrit::Runtime
{
    enum class CameraType : u8
    {
        Perspective,
        Orthographic
    };

    class IFRIT_APIDECL IF_CLASS() Camera : public Component
    {
    public:
        IF_PROPERTY(Editable, UISelect)
        CameraType mType = CameraType::Perspective;

        IF_PROPERTY(Editable, UISlider = (min = 0.1f, max = 180.0f))
        f32 mFov = 60.0f;

        IF_PROPERTY(Editable, UISlider = (min = 0.1f, max = 1000.0f))
        f32 mOrthoSpaceSize = 1.0f;

        IF_PROPERTY(Editable, UISlider = (min = 0.1f, max = 10.0f))
        f32 mAspect = 1.0f;

        IF_PROPERTY(Editable, UISlider = (min = 0.1f, max = 1000.0f))
        f32 mNear = 0.1f;

        IF_PROPERTY(Editable, UISlider = (min = 0.1f, max = 10000.0f))
        f32 mFar = 1000.0f;

        IF_PROPERTY(Editable, UISelect)
        bool mIsMainCamera = false;

    public:
        using Component::Component;
        virtual ~Camera() = default;

        Matrix4x4f        GetWorldToCameraMatrix() const;
        Matrix4x4f        GetProjectionMatrix() const;
        Vector4f          GetFront() const;

        // getters

        inline f32        GetFov() const { return mFov; }
        inline f32        GetAspect() const { return mAspect; }
        inline f32        GetNear() const { return mNear; }
        inline f32        GetFar() const { return mFar; }
        inline bool       GetIsMainCamera() const { return mIsMainCamera; }
        inline f32        GetOrthoSpaceSize() const { return mOrthoSpaceSize; }
        inline CameraType GetCameraType() const { return mType; }

        // setters

        inline void       SetFov(f32 fov) { mFov = fov; }
        inline void       SetAspect(f32 aspect) { mAspect = aspect; }
        inline void       SetNear(f32 nearx) { mNear = nearx; }
        inline void       SetFar(f32 farx) { mFar = farx; }
        inline void       SetMainCamera(bool isMain) { mIsMainCamera = isMain; }
        inline void       SetOrthoSpaceSize(f32 size) { mOrthoSpaceSize = size; }
        inline void       SetCameraType(CameraType type) { mType = type; }
    };
} // namespace Ifrit::Runtime
