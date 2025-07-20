
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
#include "ifrit/runtime/common/Pch.h"

#include "ifrit/runtime/base/Camera.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/util/TimingRecorder.h"
#include "ifrit/runtime/scene/FrameCollector.h"

namespace Ifrit::Runtime
{
    class Scene;

    class IFRIT_APIDECL SceneNode
    {
    protected:
        Scene*                   m_parentScene;
        Vec<Ref<SceneNode>>      m_children;
        Vec<GameObject*>         m_GameObjects;
        Vec<GameObjectReference> m_GameObjectRefs;

    public:
        SceneNode();
        SceneNode(Scene* parentScene) : m_parentScene(parentScene){};
        virtual ~SceneNode() = default;
        Ref<SceneNode>             AddChildNode();
        GameObject*                AddGameObject(const String& name);
        GameObject*                AddGameObjectTransferred(GameObject* obj);

        inline Ref<SceneNode>      GetSceneNode(u32 x) { return m_children.at(x); }
        inline GameObject*         GetGameObject(u32 x) { return m_GameObjects.at(x); }
        inline Vec<Ref<SceneNode>> GetChildren()
        {
            Vec<Ref<SceneNode>> x;
            for (auto& y : m_children)
            {
                x.push_back(y);
            }
            return x;
        }
        inline Vec<GameObject*> GetGameObjects() { return m_GameObjects; }

        void                    OnComponentStart();
        void                    OnComponentAwake();
        void                    OnUpdate();
        void                    OnFixedUpdate();

        IFRIT_STRUCT_SERIALIZE(m_children, m_GameObjectRefs);
    };

    class IFRIT_APIDECL Scene : public IComponentManagerKeeper
    {
    protected:
        Ref<ComponentManager>  m_componentManager;  // This dtor should be called in order
        Ref<GameObjectManager> m_gameObjectManager; // This dtor should be called in order

        Ref<SceneNode>         m_root;
        bool                   m_isAwake       = false;
        u64                    m_curFixedFrame = 0;
        Ref<PerFrameData>      m_perFrameData;

    public:
        Scene();

        inline Ref<SceneNode> GetRootNode() { return m_root; }
        Camera*               GetMainCamera();

        Ref<SceneNode>        AddSceneNode();
        Vec<GameObject*>      FilterObjects(Fn<bool(GameObject*)> filter);

        void                  OnComponentStart();
        void                  OnComponentAwake();
        void                  OnUpdate();
        void                  OnFixedUpdate(TimingRecorder* stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames);

        void                  InvokeFrameUpdate();

        Ref<PerFrameData>     GetPerFrameData() { return m_perFrameData; }

        inline ComponentManager*  GetComponentManager() override { return m_componentManager.get(); }
        inline GameObjectManager* GetGameObjectManager() override { return m_gameObjectManager.get(); }

        void                      DepthFirstTraverse(
                                 Fn<bool(SceneNode*)> fnNode, Fn<void(GameObject*)> fnObject, Fn<void()> fnOnPush, Fn<void()> fnOnPop);
        IFRIT_STRUCT_SERIALIZE(m_root);
    };

} // namespace Ifrit::Runtime