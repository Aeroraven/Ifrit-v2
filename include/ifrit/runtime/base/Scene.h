
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
#include "ifrit/core/reflection/ReflAttrs.h"

namespace Ifrit::Runtime
{
    class Scene;
    using SceneNodeId = u32;

    class IFRIT_APIDECL IF_CLASS() SceneNode
    {
    public:
        IF_PROPERTY()
        String mName;

        IF_PROPERTY()
        GUID mGuid;

        IF_PROPERTY()
        Vec<SceneNodeId> mChildren;

        IF_PROPERTY()
        Vec<GameObjectReference> mGameObjectRefs;

    protected:
        Scene*           m_Parent;
        Vec<GameObject*> m_GameObjects;

    public:
        SceneNode();
        SceneNode(Scene* parentScene) : m_Parent(parentScene) {};
        virtual ~SceneNode() = default;

        SceneNode*              AddChildNode(const String& name);
        GameObject*             AddGameObject(const String& name);
        GameObject*             AddGameObjectGPUTransform(const String& name);
        GameObject*             AddGameObjectTransferred(GameObject* obj);

        SceneNode*              GetSceneNode(u32 x);
        inline GameObject*      GetGameObject(u32 x) { return m_GameObjects.at(x); }
        Vec<SceneNode*>         GetChildren();
        inline Vec<GameObject*> GetGameObjects() { return m_GameObjects; }
        inline GUID             GetGUID() const { return mGuid; }
        inline String           GetName() const { return mName; }

        void                    OnComponentStart();
        void                    OnComponentAwake();
        void                    OnUpdate();
        void                    OnFixedUpdate();

        friend class Scene;
    };

    class IFRIT_APIDECL IF_CLASS() Scene : public IComponentManagerKeeper
    {
    public:
        // Note:This dtor should be called in order

        IF_PROPERTY()
        Owner<ComponentManager> mComponentManager;

        IF_PROPERTY()
        Owner<GameObjectManager> mGameObjectManager;

        IF_PROPERTY()
        Vec<Owner<SceneNode>> mSceneNodes;

        IF_PROPERTY()
        Owner<SceneNode> mRoot;

    protected:
        Owner<PerFrameData> m_PerFrameData;
        bool                m_IsAwake       = false;
        u64                 m_CurFixedFrame = 0;

    public:
        Scene();

        SceneNode*         GetRootNode();
        Camera*            GetMainCamera();

        u32                AllocateSceneNode(const String& name);
        SceneNode*         GetSceneNode(u32 id);

        SceneNode*         AddSceneNode(const String& name);
        Vec<GameObject*>   FilterObjects(Fn<bool(GameObject*)> filter);
        Vec<SceneNode*>    FilterNodes(Fn<bool(SceneNode*)> filter);

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

        String Serialize() const;
    };

} // namespace Ifrit::Runtime