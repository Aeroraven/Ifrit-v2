
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
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/reflection/Reflection.h"
#include "ifrit/core/reflection/SerializeHelper.h"

#include "ifrit/runtime/asset/util/PrefabSerializer.h"
using namespace Ifrit::Math;

namespace Ifrit::Runtime
{

    IFRIT_APIDECL Component::Component(GameObject* parent)
    {
        mGuid               = GUID::Generate();
        m_ParentRef         = parent->GetManagerId();
        m_GameObjectManager = parent->m_GameObjectManager;
    }

    IFRIT_APIDECL void GameObject::Initialize(ComponentManager* manager, GameObjectManager* gameObjectManager)
    {
        m_ComponentManager  = manager;
        m_GameObjectManager = gameObjectManager;
        AddComponent<Transform>();
    }

    IFRIT_APIDECL GameObject::GameObject() { mGuid = GUID::Generate(); }

    IFRIT_APIDECL GameObject::~GameObject()
    {
        for (auto& [typeHash, index] : mComponentsHashed)
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
        bool last = mEnabled;
        mEnabled  = enable;
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
        auto& auxHandles = Reflection::GetPropertyUIAuxHandles();
        auto  object     = Reflection::ReferenceObject(this);
        auto  uiHandles  = Reflection::GetPropertyEditorHandles(object);
        for (auto& handle : uiHandles)
        {
            if (auxHandles.m_OnPreRegister)
            {
                auxHandles.m_OnPreRegister();
            }
            handle();
            if (auxHandles.m_OnPostRegister)
            {
                auxHandles.m_OnPostRegister();
            }
        }
    }

    IFRIT_APIDECL void Component::CallFunctionEditorHandle()
    {
        auto object    = Reflection::ReferenceObject(this);
        auto uiHandles = Reflection::GetMethodEditorHandles(object);
        for (auto& handle : uiHandles)
        {
            handle();
        }
    }

    IFRIT_APIDECL u32 Component::GetNumVisibleProperties() const
    {
        auto object = Reflection::ReferenceObject(const_cast<Component*>(this));
        return SizeCast<u32>(Reflection::GetNumVisibleProperties(object));
    }

    IFRIT_APIDECL      ComponentManager::ComponentManager() {}

    IFRIT_APIDECL void ComponentManager::RequestRemove(Component* component)
    {

        auto  typeHash                                        = mIdToTypeHash[component->GetManagedIndex()];
        auto& tailCom                                         = mComponentArray[typeHash].back();
        tailCom->mArrayIndex                                  = component->GetArrayIndex();
        mComponentArray[typeHash][component->GetArrayIndex()] = std::move(tailCom);
        mComponentArray[typeHash].pop_back();
        // Release id
        mAllocatedComponents--;
    }

    IFRIT_APIDECL u32 ComponentManager::AllocateId()
    {
        if (m_FreeIdQueue.empty())
        {
            return mAllocatedComponents++;
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
        auto id                  = AllocateId();
        component->mArrayIndex   = arrayPos;
        component->mManagedIndex = id;
        mIdToTypeHash[id]        = typeHash;
    }

    IFRIT_APIDECL ComponentReference ComponentManager::CreateComponentFromMeta(
        GameObject* parentObject, const FMetaTypeInfo& metaTypeInfo, bool enabled)
    {
        auto typeHash = metaTypeInfo.Hash;
        if (mComponentArray.count(typeHash) == 0)
        {
            mComponentArray[typeHash] = Vec<Owner<Component>>();
        }
        auto             retd = Reflection::ConstructObject(metaTypeInfo);
        Owner<Component> ret;
        retd.ObjectValue.ForcedReinterpretTransferTo(ret);
        ret->mEnabled = enabled;
        SetComponentId(ret.get(), SizeCast<u32>(mComponentArray[typeHash].size()), typeHash);
        mComponentArray[typeHash].push_back(std::move(ret));
        return { typeHash, SizeCast<u32>(mComponentArray[typeHash].size() - 1) };
    }

    // GameObjectManager
    IFRIT_APIDECL u32 GameObjectManager::AllocateId()
    {
        if (m_FreeIdQueue.empty())
        {
            return mAllocatedObjects++;
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
        mGameObjects.push_back(std::move(gameObject));
        return id;
    }
    IFRIT_APIDECL void GameObjectManager::RebuildLookupTable()
    {
        m_GameObjectNameToIndex.clear();
        m_GameObjectUUIDToIndex.clear();
        for (u32 i = 0; i < mGameObjects.size(); ++i)
        {
            auto& gameObject = mGameObjects[i];
            if (gameObject)
            {
                m_GameObjectNameToIndex[gameObject->GetName()] = i;
                m_GameObjectUUIDToIndex[gameObject->GetUUID()] = i;
            }
        }
    }

    IFRIT_APIDECL GameObject* GameObjectManager::GetGameObject(GameObjectReference ref)
    {
        if (ref >= mGameObjects.size())
        {
            IF_LOG_CRITICAL("GameObjectManager", "Invalid GameObject reference: {}", ref);
            return nullptr;
        }
        return mGameObjects[ref].get();
    }

    IFRIT_APIDECL void GameObject::AddComponentFromeMeta(const FMetaTypeInfo& metaTypeInfo, bool enabled)
    {
        auto componentRef = m_ComponentManager->CreateComponentFromMeta(this, metaTypeInfo, enabled);
        auto typeHash     = metaTypeInfo.Hash;
        if (mComponentsHashed.count(typeHash) > 0)
        {
            IF_LOG_CRITICAL("GameObject", "Component type name conflicted");
            std::abort();
        }
        mComponentsHashed[typeHash] = componentRef.second;
    }

    Owner<Prefab> GameObject::CreatePrefab()
    {
        Owner<Prefab>               prefab = MakeOwner<Prefab>();
        TempPrefabSerializationData tempData;
        tempData.mGameObject = Owner<GameObject>(this);
        auto components      = GetAllComponents();
        for (auto& component : components)
        {
            tempData.mComponents.push_back(Owner<Component>(component));
        }
        String serializedData   = Reflection::SerializeToJSON(tempData);
        prefab->mSerializedData = serializedData;
        tempData.mGameObject.release();
        for (auto& component : tempData.mComponents)
        {
            component.release();
        }
        tempData.mComponents.clear();
        return prefab;
    }

} // namespace Ifrit::Runtime