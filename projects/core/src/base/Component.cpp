
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

#include "ifrit/core/base/Component.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/Identifier.h"
#include <atomic>
#include <random>

using namespace Ifrit::Math;

namespace Ifrit::Core
{

    IFRIT_APIDECL Ref<SceneObject> SceneObject::CreatePrefab()
    {
        auto prefab = std::make_shared<SceneObjectPrefab>();
        prefab->Initialize();
        prefab->AddComponent<Transform>();
        return prefab;
    }

    IFRIT_APIDECL
    Component::Component(Ref<SceneObject> parent)
        : m_parentObject(parent), m_parentObjectRaw(parent.get())
    {
        Ifrit::Common::Utility::GenerateUuid(m_id.m_uuid);
    }
    IFRIT_APIDECL void SceneObject::Initialize()
    {
        AddComponent<Transform>();
    }
    IFRIT_APIDECL SceneObject::SceneObject()
    {
        Ifrit::Common::Utility::GenerateUuid(m_id.m_uuid);
    }

    IFRIT_APIDECL void Component::SetEnable(bool enable)
    {
        bool last   = m_isEnabled;
        m_isEnabled = enable;
        if (!last && enable)
        {
            m_shouldInvokeStart = true;
        }
    }

    IFRIT_APIDECL void Component::InvokeStart()
    {
        if (m_shouldInvokeStart)
        {
            OnStart();
            m_shouldInvokeStart = false;
        }
    }

    IFRIT_APIDECL void Component::InvokeAwake()
    {
        if (m_shouldInvokeAwake)
        {
            OnAwake();
            m_shouldInvokeAwake = false;
        }
    }

    IFRIT_APIDECL Matrix4x4f Transform::GetModelToWorldMatrix()
    {
        Matrix4x4f model = Identity4();
        model            = MatMul(Scale3D(m_attributes.m_scale), model);
        model            = MatMul(EulerAngleToMatrix(m_attributes.m_rotation), model);
        model            = MatMul(Translate3D(m_attributes.m_position), model);
        return model;
    }

    IFRIT_APIDECL Matrix4x4f Transform::GetModelToWorldMatrixLast()
    {
        Matrix4x4f model = Identity4();
        model            = MatMul(Scale3D(m_lastFrame.m_scale), model);
        model            = MatMul(EulerAngleToMatrix(m_lastFrame.m_rotation), model);
        model            = MatMul(Translate3D(m_lastFrame.m_position), model);
        return model;
    }

} // namespace Ifrit::Core