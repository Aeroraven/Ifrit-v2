
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
#include "ifrit/core/base/IfritBase.h"
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
        Scene*               m_parentScene;
        Vec<Ref<SceneNode>>  m_children;
        Vec<Ref<GameObject>> m_gameObjects;

    public:
        SceneNode();
        SceneNode(Scene* parentScene) : m_parentScene(parentScene){};
        virtual ~SceneNode() = default;
        Ref<SceneNode>             AddChildNode();
        Ref<GameObject>            AddGameObject(const String& name);
        Ref<GameObject>            AddGameObjectTransferred(Ref<GameObject>&& obj);

        inline Ref<SceneNode>      GetSceneNode(u32 x) { return m_children.at(x); }
        inline Ref<GameObject>     GetGameObject(u32 x) { return m_gameObjects.at(x); }
        inline Vec<Ref<SceneNode>> GetChildren()
        {
            Vec<Ref<SceneNode>> x;
            for (auto& y : m_children)
            {
                x.push_back(y);
            }
            return x;
        }
        inline Vec<Ref<GameObject>> GetGameObjects()
        {
            Vec<Ref<GameObject>> x;
            for (auto& y : m_gameObjects)
            {
                x.push_back(y);
            }
            return x;
        }

        void OnComponentStart();
        void OnComponentAwake();
        void OnUpdate();
        void OnFixedUpdate();

        IFRIT_STRUCT_SERIALIZE(m_children, m_gameObjects);
    };

    class IFRIT_APIDECL Scene : public IComponentManagerKeeper
    {
    protected:
        Ref<ComponentManager> m_componentManager; // This dtor should be called last
        Ref<SceneNode>        m_root;
        bool                  m_isAwake       = false;
        u64                   m_curFixedFrame = 0;
        Ref<PerFrameData>     m_perFrameData;

    public:
        Scene();

        inline Ref<SceneNode> GetRootNode() { return m_root; }
        Camera*               GetMainCamera();

        Ref<SceneNode>        AddSceneNode();
        Vec<Ref<GameObject>>  FilterObjects(Fn<bool(Ref<GameObject>)> filter);
        Vec<GameObject*>      FilterObjectsUnsafe(Fn<bool(GameObject*)> filter);

        void                  OnComponentStart();
        void                  OnComponentAwake();
        void                  OnUpdate();
        void                  OnFixedUpdate(TimingRecorder* stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames);

        void                  InvokeFrameUpdate();

        Ref<PerFrameData>     GetPerFrameData() { return m_perFrameData; }

        inline ComponentManager* GetComponentManager() override { return m_componentManager.get(); }
        IFRIT_STRUCT_SERIALIZE(m_root);
    };

} // namespace Ifrit::Runtime