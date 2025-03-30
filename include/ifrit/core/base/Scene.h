
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/core/base/Camera.h"
#include "ifrit/core/base/Component.h"
#include "ifrit/core/util/TimingRecorder.h"
#include "ifrit/core/scene/FrameCollector.h"

namespace Ifrit::Core
{

    class IFRIT_APIDECL SceneNode
    {
    protected:
        Vec<Ref<SceneNode>>   m_children;
        Vec<Ref<SceneObject>> m_gameObjects;

    public:
        SceneNode()          = default;
        virtual ~SceneNode() = default;
        Ref<SceneNode>             AddChildNode();
        Ref<SceneObject>           AddGameObject(const String& name);
        Ref<SceneObject>           AddGameObjectTransferred(Ref<SceneObject>&& obj);

        inline Ref<SceneNode>      GetSceneNode(u32 x) { return m_children.at(x); }
        inline Ref<SceneObject>    GetGameObject(u32 x) { return m_gameObjects.at(x); }
        inline Vec<Ref<SceneNode>> GetChildren()
        {
            Vec<Ref<SceneNode>> x;
            for (auto& y : m_children)
            {
                x.push_back(y);
            }
            return x;
        }
        inline Vec<Ref<SceneObject>> GetGameObjects()
        {
            Vec<Ref<SceneObject>> x;
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

    class IFRIT_APIDECL Scene
    {
    protected:
        Ref<SceneNode>    m_root;
        bool              m_isAwake       = false;
        u64               m_curFixedFrame = 0;
        Ref<PerFrameData> m_perFrameData;

    public:
        Scene();

        inline Ref<SceneNode> GetRootNode() { return m_root; }
        Camera*               GetMainCamera();

        Ref<SceneNode>        AddSceneNode();
        Vec<Ref<SceneObject>> FilterObjects(Fn<bool(Ref<SceneObject>)> filter);
        Vec<SceneObject*>     FilterObjectsUnsafe(Fn<bool(SceneObject*)> filter);

        void                  OnComponentStart();
        void                  OnComponentAwake();
        void                  OnUpdate();
        void                  OnFixedUpdate(TimingRecorder* stopwatch, u32 fixedUpdateRate, u32 maxCompensationFrames);

        void                  InvokeFrameUpdate();

        Ref<PerFrameData>     GetPerFrameData() { return m_perFrameData; }
        IFRIT_STRUCT_SERIALIZE(m_root);
    };

} // namespace Ifrit::Core