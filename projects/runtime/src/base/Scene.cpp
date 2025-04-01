
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
    IFRIT_APIDECL SceneNode::SceneNode() : m_parentScene(nullptr)
    {
        iWarn("SceneNode constructor called without parent scene! Serialization system is under development.");
    }

    IFRIT_APIDECL Ref<SceneNode> SceneNode::AddChildNode()
    {
        auto node = std::make_shared<SceneNode>(m_parentScene);
        m_children.push_back(node);
        return node;
    }
    IFRIT_APIDECL Ref<GameObject> SceneNode::AddGameObject(const String& name)
    {
        auto obj = std::make_shared<GameObject>();
        obj->Initialize(m_parentScene->GetComponentManager());
        obj->SetName(name);
        m_gameObjects.push_back(obj);
        return obj;
    }

    IFRIT_APIDECL Ref<GameObject> SceneNode::AddGameObjectTransferred(Ref<GameObject>&& obj)
    {
        m_gameObjects.push_back(std::move(obj));
        return obj;
    }

    IFRIT_APIDECL void SceneNode::OnUpdate()
    {
        for (auto& child : m_children)
        {
            child->OnUpdate();
        }
        for (auto& obj : m_gameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                comp->OnUpdate();
            }
        }
    }

    IFRIT_APIDECL void SceneNode::OnComponentStart()
    {
        for (auto& child : m_children)
        {
            child->OnComponentStart();
        }
        for (auto& obj : m_gameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                comp->InvokeStart();
            }
        }
    }

    IFRIT_APIDECL void SceneNode::OnComponentAwake()
    {
        for (auto& child : m_children)
        {
            child->OnComponentAwake();
        }
        for (auto& obj : m_gameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                comp->InvokeAwake();
            }
        }
    }

    IFRIT_APIDECL void SceneNode::OnFixedUpdate()
    {
        for (auto& child : m_children)
        {
            child->OnFixedUpdate();
        }
        for (auto& obj : m_gameObjects)
        {
            for (auto& comp : obj->GetAllComponents())
            {
                comp->OnFixedUpdate();
            }
        }
    }

    IFRIT_APIDECL Ref<SceneNode> Scene::AddSceneNode() { return m_root->AddChildNode(); }

    IFRIT_APIDECL Camera*        Scene::GetMainCamera()
    {
        Vec<SceneNode*> nodes;
        nodes.push_back(m_root.get());
        while (!nodes.empty())
        {
            auto node = nodes.back();
            nodes.pop_back();
            for (auto& child : node->GetChildren())
            {
                nodes.push_back(child.get());
            }
            for (auto& obj : node->GetGameObjects())
            {
                auto camera = obj->GetComponent<Camera>();
                if (camera)
                {
                    if (camera->IsMainCamera())
                        return camera.get();
                }
            }
        }
        return nullptr;
    }

    IFRIT_APIDECL Vec<Ref<GameObject>> Scene::FilterObjects(Fn<bool(Ref<GameObject>)> filter)
    {
        Vec<Ref<GameObject>> result;
        Vec<SceneNode*>      nodes;
        nodes.push_back(m_root.get());
        while (!nodes.empty())
        {
            auto node = nodes.back();
            nodes.pop_back();
            for (auto& child : node->GetChildren())
            {
                nodes.push_back(child.get());
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

    IFRIT_APIDECL Vec<GameObject*> Scene::FilterObjectsUnsafe(Fn<bool(GameObject*)> filter)
    {
        Vec<GameObject*> result;
        Vec<SceneNode*>  nodes;
        nodes.push_back(m_root.get());
        while (!nodes.empty())
        {
            auto node = nodes.back();
            nodes.pop_back();
            for (auto& child : node->GetChildren())
            {
                nodes.push_back(child.get());
            }
            for (auto& obj : node->GetGameObjects())
            {
                if (filter(obj.get()))
                {
                    result.push_back(obj.get());
                }
            }
        }
        return result;
    }

    IFRIT_APIDECL void Scene::OnUpdate() { m_root->OnUpdate(); }
    IFRIT_APIDECL void Scene::OnComponentAwake() { m_root->OnComponentAwake(); }
    IFRIT_APIDECL void Scene::OnComponentStart() { m_root->OnComponentStart(); }

    IFRIT_APIDECL void Scene::OnFixedUpdate(TimingRecorder* stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames)
    {
        auto lastTimeStamp = stopwatch->GetCurTimeUs();
        auto totalFrames   = lastTimeStamp / fixedUpdateRate;
        auto sourceFrame   = m_curFixedFrame;

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
            m_root->OnFixedUpdate();
        }
        m_curFixedFrame = totalFrames;
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

    IFRIT_APIDECL Scene::Scene()
        : m_root(std::make_shared<SceneNode>(this)), m_perFrameData(std::make_shared<PerFrameData>())
    {
        m_componentManager = std::make_shared<ComponentManager>();
    }

} // namespace Ifrit::Runtime