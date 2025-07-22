
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
    using SceneNodeId = u32;

    class IFRIT_APIDECL SceneNode
    {
    protected:
        Scene*                   m_Parent;
        Vec<SceneNodeId>         m_Children;
        Vec<GameObject*>         m_GameObjects;
        Vec<GameObjectReference> m_GameObjectRefs;

    public:
        SceneNode();
        SceneNode(Scene* parentScene) : m_Parent(parentScene){};
        virtual ~SceneNode() = default;

        SceneNode*              AddChildNode();
        GameObject*             AddGameObject(const String& name);
        GameObject*             AddGameObjectTransferred(GameObject* obj);

        SceneNode*              GetSceneNode(u32 x);
        inline GameObject*      GetGameObject(u32 x) { return m_GameObjects.at(x); }
        Vec<SceneNode*>         GetChildren();
        inline Vec<GameObject*> GetGameObjects() { return m_GameObjects; }

        void                    OnComponentStart();
        void                    OnComponentAwake();
        void                    OnUpdate();
        void                    OnFixedUpdate();

        IFRIT_STRUCT_SERIALIZE(m_Children, m_GameObjectRefs);
    };

    class IFRIT_APIDECL Scene : public IComponentManagerKeeper
    {
    protected:
        Owner<ComponentManager>  m_ComponentManager;  // This dtor should be called in order
        Owner<GameObjectManager> m_GameObjectManager; // This dtor should be called in order
        Vec<Owner<SceneNode>>    m_SceneNodes;        // This dtor should be called in order
        Owner<SceneNode>         m_Root;
        Owner<PerFrameData>      m_PerFrameData;
        bool                     m_IsAwake       = false;
        u64                      m_CurFixedFrame = 0;

    public:
        Scene();

        SceneNode*         GetRootNode();
        Camera*            GetMainCamera();

        u32                AllocateSceneNode();
        SceneNode*         GetSceneNode(u32 id);

        SceneNode*         AddSceneNode();
        Vec<GameObject*>   FilterObjects(Fn<bool(GameObject*)> filter);

        void               OnComponentStart();
        void               OnComponentAwake();
        void               OnUpdate();
        void               OnFixedUpdate(TimingRecorder* stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames);

        void               InvokeFrameUpdate();

        PerFrameData*      GetPerFrameData();

        ComponentManager*  GetComponentManager() override;
        GameObjectManager* GetGameObjectManager() override;

        void               DepthFirstTraverse(
                          Fn<bool(SceneNode*)> fnNode, Fn<void(GameObject*)> fnObject, Fn<void()> fnOnPush, Fn<void()> fnOnPop);
        IFRIT_STRUCT_SERIALIZE(m_Root);
    };

} // namespace Ifrit::Runtime