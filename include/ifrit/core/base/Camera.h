
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
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"

#include "ifrit/common/util/TypingUtil.h"

namespace Ifrit::Core
{
    enum class CameraType
    {
        Perspective,
        Orthographic
    };
    struct CameraData
    {
        CameraType m_type           = CameraType::Perspective; // Ortho support not implemented
        f32        m_fov            = 60.0f;
        f32        m_orthoSpaceSize = 1.0f;
        f32        m_aspect         = 1.0f;
        f32        m_near           = 0.1f;
        f32        m_far            = 1000.0f;
        bool       m_IsMainCamera   = false;
        IFRIT_STRUCT_SERIALIZE(m_type, m_fov, m_aspect, m_near, m_far, m_IsMainCamera);
    };
    class IFRIT_APIDECL Camera : public Component, public AttributeOwner<CameraData>
    {
    public:
        Camera(){};
        Camera(std::shared_ptr<SceneObject> owner)
            : Component(owner), AttributeOwner() {}
        virtual ~Camera() = default;
        inline std::string Serialize() override { return SerializeAttribute(); }
        inline void        Deserialize() override { DeserializeAttribute(); }
        Matrix4x4f         GetWorldToCameraMatrix() const;
        Matrix4x4f         GetProjectionMatrix() const;
        Vector4f           GetFront() const;

        // getters
        inline f32         GetFov() const { return m_attributes.m_fov; }
        inline f32         GetAspect() const { return m_attributes.m_aspect; }
        inline f32         GetNear() const { return m_attributes.m_near; }
        inline f32         GetFar() const { return m_attributes.m_far; }
        inline bool        IsMainCamera() const { return m_attributes.m_IsMainCamera; }
        inline f32         GetOrthoSpaceSize() const { return m_attributes.m_orthoSpaceSize; }
        inline CameraType  GetCameraType() const { return m_attributes.m_type; }

        // setters
        inline void        SetFov(f32 fov) { m_attributes.m_fov = fov; }
        inline void        SetAspect(f32 aspect) { m_attributes.m_aspect = aspect; }
        inline void        SetNear(f32 nearx) { m_attributes.m_near = nearx; }
        inline void        SetFar(f32 farx) { m_attributes.m_far = farx; }
        inline void        SetMainCamera(bool isMain) { m_attributes.m_IsMainCamera = isMain; }
        inline void        SetOrthoSpaceSize(f32 size) { m_attributes.m_orthoSpaceSize = size; }
        inline void        SetCameraType(CameraType type) { m_attributes.m_type = type; }
    };
} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Camera)
IFRIT_ENUMCLASS_SERIALIZE(Ifrit::Core::CameraType)