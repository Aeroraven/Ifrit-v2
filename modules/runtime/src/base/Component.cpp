
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
#include "ifrit/core/math/linalg/LinalgOps.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/core/algo/GUID.h"
#include <atomic>
#include <random>

using namespace Ifrit::Math;

namespace Ifrit::Runtime
{

    // TODO: REMOVING THIS FUNCTION
    IFRIT_APIDECL GameObject* GameObject::CreatePrefab(IComponentManagerKeeper* managerKeeper)
    {
        auto prefabIdx = managerKeeper->GetGameObjectManager()->CreateGameObject("Prefab");
        auto prefab    = managerKeeper->GetGameObjectManager()->GetGameObject(prefabIdx);
        prefab->Initialize(managerKeeper->GetComponentManager(), managerKeeper->GetGameObjectManager());
        return prefab;
    }

    IFRIT_APIDECL
    Component::Component(GameObject* parent)
    {
        m_id.m_GUID         = GUID::Generate();
        m_ParentRef         = parent->GetManagerId();
        m_GameObjectManager = parent->m_GameObjectManager;
    }

    IFRIT_APIDECL void GameObject::Initialize(ComponentManager* manager, GameObjectManager* gameObjectManager)
    {
        m_ComponentManager  = manager;
        m_GameObjectManager = gameObjectManager;
        AddComponent<Transform>();
    }

    IFRIT_APIDECL GameObject::GameObject() { m_Identifier.m_GUID = GUID::Generate(); }

    IFRIT_APIDECL GameObject::~GameObject()
    {
        for (auto& [typeHash, index] : m_ComponentsHashed)
        {
            auto component = m_ComponentManager->GetComponentFromReference<Component>({ typeHash, index });
            if (component)
            {
                component->OnEnd();
                m_ComponentManager->RequestRemove(component);
            }
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

    IFRIT_APIDECL GameObject* Component::GetParent() const
    {
        if (m_GameObjectManager)
        {
            return m_GameObjectManager->GetGameObject(m_ParentRef);
        }
        IF_LOG_CRITICAL("Component", "GameObjectManager is not set for this component");
        return nullptr;
    }
    IFRIT_APIDECL void Component::InvokeAwake()
    {
        if (m_shouldInvokeAwake)
        {
            OnAwake();
            m_shouldInvokeAwake = false;
        }
    }

    IFRIT_APIDECL void Component::CallPropertyEditorHandle()
    {
        // iDebug("Num properties: {}", m_Property.size());
        if (!m_PropertyRegistered)
        {
            m_PropertyRegistered = true;
            AddProperty<bool, EPropertyEditorType::Select>("Enable", m_isEnabled);
            SetupProperties();
        }
        auto& axuHandles = GetPropertyEditorAxuHandles();
        for (auto& prop : m_Property)
        {
            if (axuHandles.m_OnPreRegister)
            {
                axuHandles.m_OnPreRegister();
            }
            prop.RegisterEditorHandle();
            if (axuHandles.m_OnPostRegister)
            {
                axuHandles.m_OnPostRegister();
            }
        }
    }

    IFRIT_APIDECL      ComponentManager::ComponentManager() {}

    IFRIT_APIDECL void ComponentManager::RequestRemove(Component* component)
    {

        auto  meta                                    = component->GetMetaData();
        auto  typeHash                                = m_IdToTypeHash[meta.m_ManagerIndex];
        auto& tailCom                                 = m_ComponentArray[typeHash].back();
        tailCom->m_id.m_ArrayIndex                    = meta.m_ArrayIndex;
        m_ComponentArray[typeHash][meta.m_ArrayIndex] = std::move(tailCom);
        m_ComponentArray[typeHash].pop_back();
        // Release id
        m_AllocatedComponents--;
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

    IFRIT_APIDECL void ComponentManager::SetComponentId(Component* component, u32 arrayPos, ComponentTypeHash typeHash)
    {
        auto id                        = AllocateId();
        component->m_id.m_ArrayIndex   = arrayPos;
        component->m_id.m_ManagerIndex = id;
        m_IdToTypeHash[id]             = typeHash;
    }

    // GameObjectManager
    IFRIT_APIDECL u32 GameObjectManager::AllocateId()
    {
        if (m_FreeIdQueue.empty())
        {
            return m_AllocatedObjects++;
        }
        else
        {
            auto id = m_FreeIdQueue.front();
            m_FreeIdQueue.pop();
            return id;
        }
    }

    IFRIT_APIDECL GameObjectReference GameObjectManager::CreateGameObject(const String& name)
    {
        auto id = AllocateId();
        if (m_GameObjectNameToIndex.count(name) > 0)
        {
            IF_LOG_CRITICAL("GameObjectManager", "GameObject name already exists: {}", name);
        }
        m_GameObjectNameToIndex[name] = id;

        auto gameObject = MakeOwner<GameObject>();
        gameObject->SetName(name);
        gameObject->SetManagerId(id);
        m_GameObjectUUIDToIndex[gameObject->GetUUID()] = id;
        m_GameObjects.push_back(std::move(gameObject));
        return id;
    }

    IFRIT_APIDECL GameObject* GameObjectManager::GetGameObject(GameObjectReference ref)
    {
        if (ref >= m_GameObjects.size())
        {
            IF_LOG_CRITICAL("GameObjectManager", "Invalid GameObject reference: {}", ref);
            return nullptr;
        }
        return m_GameObjects[ref].get();
    }

} // namespace Ifrit::Runtime