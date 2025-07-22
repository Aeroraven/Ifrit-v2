
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

#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/base/Component.h"
namespace Ifrit::Runtime
{
    IFRIT_APIDECL            SceneNode::SceneNode() : m_Parent(nullptr) {}

    IFRIT_APIDECL SceneNode* SceneNode::AddChildNode()
    {
        auto nodeId = m_Parent->AllocateSceneNode();
        auto node   = m_Parent->GetSceneNode(nodeId);
        m_Children.push_back(nodeId);
        return node;
    }

    IFRIT_APIDECL SceneNode* SceneNode::GetSceneNode(u32 x)
    {
        if (x < m_Children.size())
        {
            return m_Parent->GetSceneNode(m_Children[x]);
        }
        IF_LOG_CRITICAL("SceneNode", "Invalid SceneNode ID: {}", x);
        return nullptr;
    }

    IFRIT_APIDECL Vec<SceneNode*> SceneNode::GetChildren()
    {
        Vec<SceneNode*> result;
        for (auto& childId : m_Children)
        {
            result.push_back(m_Parent->GetSceneNode(childId));
        }
        return result;
    }

    IFRIT_APIDECL GameObject* SceneNode::AddGameObject(const String& name)
    {
        auto objId = m_Parent->GetGameObjectManager()->CreateGameObject(name);
        auto obj   = m_Parent->GetGameObjectManager()->GetGameObject(objId);
        obj->Initialize(m_Parent->GetComponentManager(), m_Parent->GetGameObjectManager());
        obj->SetName(name);
        m_GameObjects.push_back(obj);
        m_GameObjectRefs.push_back(objId);
        return obj;
    }

    IFRIT_APIDECL GameObject* SceneNode::AddGameObjectTransferred(GameObject* obj)
    {
        auto idx = obj->GetManagerId();
        m_GameObjects.push_back(obj);
        m_GameObjectRefs.push_back(idx);
        if (idx == ~0u)
        {
            IF_LOG_WARNING("SceneNode", "GameObject transferred without valid manager ID, this may cause issues.");
        }
        return obj;
    }

    IFRIT_APIDECL void SceneNode::OnUpdate()
    {
        for (auto& childId : m_Children)
        {
            auto child = m_Parent->GetSceneNode(childId);
            child->OnUpdate();
        }
        for (auto& obj : m_GameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                if (comp->IsEnabled())
                    comp->OnUpdate();
            }
        }
    }

    IFRIT_APIDECL void SceneNode::OnComponentStart()
    {
        for (auto& childId : m_Children)
        {
            auto child = m_Parent->GetSceneNode(childId);
            child->OnComponentStart();
        }
        for (auto& obj : m_GameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                comp->InvokeStart();
            }
        }
    }

    IFRIT_APIDECL void SceneNode::OnComponentAwake()
    {
        for (auto& childId : m_Children)
        {
            auto child = m_Parent->GetSceneNode(childId);
            child->OnComponentAwake();
        }
        for (auto& obj : m_GameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                comp->InvokeAwake();
            }
        }
    }

    IFRIT_APIDECL void SceneNode::OnFixedUpdate()
    {
        for (auto& childId : m_Children)
        {
            auto child = m_Parent->GetSceneNode(childId);
            child->OnFixedUpdate();
        }
        for (auto& obj : m_GameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                if (comp->IsEnabled())
                    comp->OnFixedUpdate();
            }
        }
    }
    IFRIT_APIDECL u32 Scene::AllocateSceneNode()
    {
        auto node = MakeOwner<SceneNode>(this);
        m_SceneNodes.push_back(std::move(node));
        return static_cast<u32>(m_SceneNodes.size() - 1);
    }

    IFRIT_APIDECL SceneNode* Scene::GetSceneNode(u32 id)
    {
        if (id < m_SceneNodes.size())
        {
            return m_SceneNodes[id].get();
        }
        IF_LOG_CRITICAL("Scene", "Invalid SceneNode ID: {}", id);
        return nullptr;
    }

    IFRIT_APIDECL SceneNode*         Scene::GetRootNode() { return m_Root.get(); }

    IFRIT_APIDECL ComponentManager*  Scene::GetComponentManager() { return m_ComponentManager.get(); }

    IFRIT_APIDECL GameObjectManager* Scene::GetGameObjectManager() { return m_GameObjectManager.get(); }

    IFRIT_APIDECL SceneNode*         Scene::AddSceneNode() { return m_Root->AddChildNode(); }

    IFRIT_APIDECL Camera*            Scene::GetMainCamera()
    {
        Vec<SceneNode*> nodes;
        nodes.push_back(m_Root.get());
        while (!nodes.empty())
        {
            auto node = nodes.back();
            nodes.pop_back();
            for (auto& child : node->GetChildren())
            {
                nodes.push_back(child);
            }
            for (auto& obj : node->GetGameObjects())
            {
                auto camera = obj->GetComponent<Camera>();
                if (camera)
                {
                    if (camera->IsMainCamera())
                        return camera;
                }
            }
        }
        return nullptr;
    }

    IFRIT_APIDECL Vec<GameObject*> Scene::FilterObjects(Fn<bool(GameObject*)> filter)
    {
        Vec<GameObject*> result;
        Vec<SceneNode*>  nodes;
        nodes.push_back(m_Root.get());
        while (!nodes.empty())
        {
            auto node = nodes.back();
            nodes.pop_back();
            for (auto& child : node->GetChildren())
            {
                nodes.push_back(child);
            }
            for (auto& obj : node->GetGameObjects())
            {
                if (filter(obj))
                {
                    result.push_back(obj);
                }
            }
        }
        return result;
    }

    IFRIT_APIDECL void Scene::DepthFirstTraverse(
        Fn<bool(SceneNode*)> fnNode, Fn<void(GameObject*)> fnObject, Fn<void()> fnOnPush, Fn<void()> fnOnPop)
    {
        Fn<void(SceneNode*)> dfsFunc = [&](SceneNode* node) {
            for (auto& child : node->GetChildren())
            {
                fnOnPush();
                if (fnNode(child))
                    dfsFunc(child);
                fnOnPop();
            }
            for (auto& obj : node->GetGameObjects())
            {
                fnObject(obj);
            }
        };
        if (m_Root)
        {
            dfsFunc(m_Root.get());
        }
    }
    IFRIT_APIDECL void Scene::OnUpdate() { m_Root->OnUpdate(); }
    IFRIT_APIDECL void Scene::OnComponentAwake() { m_Root->OnComponentAwake(); }
    IFRIT_APIDECL void Scene::OnComponentStart() { m_Root->OnComponentStart(); }

    IFRIT_APIDECL void Scene::OnFixedUpdate(TimingRecorder* stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames)
    {
        auto lastTimeStamp = stopwatch->GetCurTimeUs();
        auto totalFrames   = lastTimeStamp / fixedUpdateRate;
        auto sourceFrame   = m_CurFixedFrame;

        if (sourceFrame >= totalFrames)
        {
            return;
        }
        auto framesToUpdate = totalFrames - sourceFrame;
        if (framesToUpdate > maxCompensationFrames)
        {
            framesToUpdate = maxCompensationFrames;
        }
        for (u32 i = 0; i < framesToUpdate; i++)
        {
            m_Root->OnFixedUpdate();
        }
        m_CurFixedFrame = totalFrames;
    }

    IFRIT_APIDECL void Scene::InvokeFrameUpdate()
    {
        // TODO: the awake logic here is not correct.
        // In Unity awake is called once after the system is initialized.
        // Then if the system is initialized, components dynamically added to the scene will got
        // the Awake called immediately, before GameObject.AddComponent<T> returns.
        OnComponentAwake();

        OnComponentStart();
        OnUpdate();
    }

    IFRIT_APIDECL Scene::Scene() : m_Root(MakeOwner<SceneNode>(this)), m_PerFrameData(MakeOwner<PerFrameData>())
    {
        m_ComponentManager  = MakeOwner<ComponentManager>();
        m_GameObjectManager = MakeOwner<GameObjectManager>();
    }

    IFRIT_APIDECL PerFrameData* Scene::GetPerFrameData() { return m_PerFrameData.get(); }

} // namespace Ifrit::Runtime