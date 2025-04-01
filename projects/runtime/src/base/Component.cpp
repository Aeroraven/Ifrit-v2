
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

#include "ifrit/runtime/base/Component.h"
#include "ifrit/core/math/LinalgOps.h"
#include "ifrit/core/algo/Identifier.h"
#include <atomic>
#include <random>

using namespace Ifrit::Math;

namespace Ifrit::Runtime
{

    IFRIT_APIDECL Ref<GameObject> GameObject::CreatePrefab(IComponentManagerKeeper* managerKeeper)
    {
        auto prefab = std::make_shared<GameObjectPrefab>();
        prefab->Initialize(managerKeeper->GetComponentManager());
        //prefab->AddComponent<Transform>();
        return prefab;
    }

    IFRIT_APIDECL
    Component::Component(Ref<GameObject> parent) : m_parentObject(parent), m_parentObjectRaw(parent.get())
    {
        GenerateUuid(m_id.m_uuid);
    }
    IFRIT_APIDECL void GameObject::Initialize(ComponentManager* manager)
    {
        m_componentManager = manager;
        AddComponent<Transform>();
    }
    IFRIT_APIDECL GameObject::GameObject() { GenerateUuid(m_id.m_uuid); }

    IFRIT_APIDECL GameObject::~GameObject()
    {

        for (auto& comp : m_components)
        {
            comp->OnEnd();
            m_componentManager->RequestRemove(comp.get());
        }
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

    IFRIT_APIDECL      ComponentManager::ComponentManager() {}

    IFRIT_APIDECL void ComponentManager::RequestRemove(Component* component)
    {

        auto  meta                                    = component->GetMetaData();
        auto  typeHash                                = m_IdToTypeHash[meta.m_ManagerIndex];
        auto& tailCom                                 = m_ComponentArray[typeHash].back();
        tailCom->m_id.m_ArrayIndex                    = meta.m_ArrayIndex;
        m_ComponentArray[typeHash][meta.m_ArrayIndex] = tailCom;
        m_ComponentArray[typeHash].pop_back();
        // Release id
        m_FreeIdQueue.push(meta.m_ArrayIndex);
    }

    IFRIT_APIDECL u32 ComponentManager::AllocateId()
    {
        if (m_FreeIdQueue.empty())
        {
            return m_AllocatedComponents++;
        }
        else
        {
            auto id = m_FreeIdQueue.front();
            m_FreeIdQueue.pop();
            return id;
        }
    }

    IFRIT_APIDECL void ComponentManager::SetComponentId(
        Ref<Component> component, u32 arrayPos, ComponentTypeHash typeHash)
    {
        auto id                        = AllocateId();
        component->m_id.m_ArrayIndex   = arrayPos;
        component->m_id.m_ManagerIndex = id;
        m_IdToTypeHash[id]             = typeHash;
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

} // namespace Ifrit::Runtime