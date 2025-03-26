
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
#include "AssetReference.h"
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define IFRIT_COMPONENT_SERIALIZE(...) IFRIT_STRUCT_SERIALIZE(m_id, m_parentObject, __VA_ARGS__)
#define IFRIT_COMPONENT_REGISTER(x) \
    IFRIT_DERIVED_REGISTER(x);      \
    IFRIT_INHERIT_REGISTER(Ifrit::Core::Component, x);

namespace Ifrit::Core
{

    struct ComponentIdentifier
    {
        String m_uuid;
        String m_name;

        IFRIT_STRUCT_SERIALIZE(m_uuid, m_name)
    };

    template <class T>
    class AttributeOwner
    {
    protected:
        T m_attributes{};

    public:
        inline String SerializeAttribute()
        {
            String serialized;
            Ifrit::Common::Serialization::serialize(m_attributes, serialized);
            return serialized;
        }
        inline void DeserializeAttribute()
        {
            String serialized;
            Ifrit::Common::Serialization::deserialize(serialized, m_attributes);
        }
    };

    class Component;
    class SceneObject;
    class Transform;

    // TODO: for performance considerations, components container is not consistent
    // across different build envs.

    class IFRIT_APIDECL SceneObject : public Ifrit::Common::Utility::NonCopyable, public std::enable_shared_from_this<SceneObject>
    {
    protected:
        ComponentIdentifier  m_id;
        String               m_name;
        // HashMap<String, Ref<Component>> m_components;
        Vec<Ref<Component>>  m_components;
        HashMap<String, u32> m_componentIndex;
        HashMap<size_t, u32> m_componentsHashed;

    public:
        SceneObject();
        virtual ~SceneObject() = default;
        void                    Initialize();

        static Ref<SceneObject> CreatePrefab();

        template <class T>
        void AddComponent(Ref<T> component)
        {
            using Ifrit::Common::Utility::RTypeInfo;
            static_assert(std::is_base_of<Component, T>::value, "T must be derived from Component");
            auto typeName = RTypeInfo<T>::name;
            auto typeHash = RTypeInfo<T>::hash;
            if (m_componentIndex.count(typeName) > 0)
            {
                iError("Component already exists");
                std::abort();
            }
            if (m_componentsHashed.count(typeHash) > 0)
            {
                iError("Hash conflicted");
                std::abort();
            }
            m_components.push_back(component);
            m_componentIndex[typeName]   = m_components.size() - 1;
            m_componentsHashed[typeHash] = m_components.size() - 1;
        }

        template <class T>
        Ref<T> AddComponent()
        {
            static_assert(std::is_base_of<Component, T>::value, "T must be derived from Component");
            using Ifrit::Common::Utility::RTypeInfo;
            auto typeName = RTypeInfo<T>::name;
            auto typeHash = RTypeInfo<T>::hash;
            if (m_componentIndex.count(typeName) > 0)
            {
                auto idx = m_componentIndex[typeName];
                return std::static_pointer_cast<T>(m_components[idx]);
            }
            if (m_componentsHashed.count(typeHash) > 0)
            {
                iError("Hash conflicted");
                std::abort();
            }
            auto ret = std::make_shared<T>(shared_from_this());
            m_components.push_back(ret);
            m_componentIndex[typeName]   = m_components.size() - 1;
            m_componentsHashed[typeHash] = m_components.size() - 1;
            return ret;
        }

        template <class T>
        Ref<T> GetComponent()
        {
            static_assert(std::is_base_of<Component, T>::value, "T must be derived from Component");
            using Ifrit::Common::Utility::RTypeInfo;
            auto typeHash = RTypeInfo<T>::hash;
            if (m_componentsHashed.count(typeHash) == 0)
            {
                return nullptr;
            }
            auto itIndex = m_componentsHashed[typeHash];
#ifdef _DEBUG
            auto ret = std::dynamic_pointer_cast<T>(m_components[itIndex]);
            if (ret == nullptr)
            {
                iError("Invalid cast");
                std::abort();
            }
            return ret;
#else
            return std::static_pointer_cast<T>(m_components[itIndex]);
#endif
        }

        // Unsafe version of GetComponent, use with caution. It's intended to be used
        // in performance-critical code.
        template <class T>
        T* GetComponentUnsafe()
        {
            static_assert(std::is_base_of<Component, T>::value, "T must be derived from Component");
            using Ifrit::Common::Utility::RTypeInfo;
            auto typeHash = RTypeInfo<T>::hash;
            if (m_componentsHashed.count(typeHash) == 0)
            {
                return nullptr;
            }
            auto itIndex = m_componentsHashed[typeHash];
            return static_cast<T*>(m_components[itIndex].get());
        }

        inline Vec<Ref<Component>> GetAllComponents() { return m_components; }

        inline void                SetName(const String& name) { m_name = name; }
        IFRIT_STRUCT_SERIALIZE(m_id, m_name, m_components, m_componentIndex, m_componentsHashed);
    };

    // Here, SceneObjectPrefab is a type alias for SceneObject.
    // It only indicates that the object does not belong to the scene.
    using SceneObjectPrefab = SceneObject;

    class IFRIT_APIDECL Component : public Ifrit::Common::Utility::NonCopyable
    {
    protected:
        ComponentIdentifier        m_id;
        std::weak_ptr<SceneObject> m_parentObject;
        bool                       m_isEnabled         = true;
        bool                       m_shouldInvokeStart = true;
        bool                       m_shouldInvokeAwake = true;

    private:
        SceneObject* m_parentObjectRaw = nullptr;

    public:
        Component(){}; // for deserializatioin
        Component(Ref<SceneObject> parentObject);
        virtual ~Component() = default;

        virtual String               Serialize()   = 0;
        virtual void                 Deserialize() = 0;

        virtual void                 OnFrameCollecting() {}
        virtual void                 OnAwake() {}
        virtual void                 OnStart() {}
        virtual void                 OnFixedUpdate() {}
        virtual void                 OnUpdate() {}
        virtual void                 OnEnd() {}

        inline void                  SetName(const String& name) { m_id.m_name = name; }
        virtual void                 SetAssetReferencedAttributes(const Vec<Ref<IAssetCompatible>>& out) {}
        void                         SetEnable(bool enable);

        inline String                GetName() const { return m_id.m_name; }
        inline String                GetUuid() const { return m_id.m_uuid; }
        inline Ref<SceneObject>      GetParent() const { return m_parentObject.lock(); }
        virtual Vec<AssetReference*> GetAssetRefs() { return {}; }
        inline bool                  IsEnabled() const { return m_isEnabled; }

        void                         InvokeStart();
        void                         InvokeAwake();

        // This function is intended to be used in performance-critical code.
        // Use with caution.
        inline SceneObject*          GetParentUnsafe() { return m_parentObjectRaw; }

        IFRIT_STRUCT_SERIALIZE(m_id, m_parentObject);
    };

    struct TransformAttribute
    {
        Vector3f m_position = Vector3f{ 0.0f, 0.0f, 0.0f };
        Vector3f m_rotation = Vector3f{ 0.0f, 0.0f, 0.0f };
        Vector3f m_scale    = Vector3f{ 1.0f, 1.0f, 1.0f };

        IFRIT_STRUCT_SERIALIZE(m_position, m_rotation, m_scale);
    };

    class IFRIT_APIDECL Transform : public Component, public AttributeOwner<TransformAttribute>
    {
    private:
        using GPUUniformBuffer                     = Ifrit::Graphics::Rhi::RhiMultiBuffer;
        using GPUBindId                            = Ifrit::Graphics::Rhi::RhiDescHandleLegacy;
        Ref<GPUUniformBuffer> m_gpuBuffer          = nullptr;
        Ref<GPUUniformBuffer> m_gpuBufferLast      = nullptr;
        Ref<GPUBindId>        m_gpuBindlessRef     = nullptr;
        Ref<GPUBindId>        m_gpuBindlessRefLast = nullptr;
        TransformAttribute    m_lastFrame;

        struct DirtyFlag
        {
            bool changed     = true;
            bool lastChanged = true;
        } m_dirty;

    public:
        Transform(){};
        Transform(Ref<SceneObject> parent)
            : Component(parent), AttributeOwner<TransformAttribute>() {}

        String      Serialize() override { return SerializeAttribute(); }
        void        Deserialize() override { DeserializeAttribute(); }

        inline void OnFrameCollecting()
        {
            if (m_dirty.changed)
            {
                m_lastFrame = m_attributes;
            }
            m_dirty.lastChanged = m_dirty.changed;
            m_dirty.changed     = false;
        }

        // getters
        inline Vector3f GetPosition() const { return m_attributes.m_position; }
        inline Vector3f GetRotation() const { return m_attributes.m_rotation; }
        inline Vector3f GetScale() const { return m_attributes.m_scale; }

        // setters
        inline void     SetPosition(const Vector3f& pos)
        {
            m_attributes.m_position = pos;
            m_dirty.changed         = true;
        }
        inline void SetRotation(const Vector3f& rot)
        {
            m_attributes.m_rotation = rot;
            m_dirty.changed         = true;
        }
        inline void SetScale(const Vector3f& scale)
        {
            m_attributes.m_scale = scale;
            m_dirty.changed      = true;
        }

        inline void      markUnchanged() { m_dirty.changed = false; }

        inline DirtyFlag GetDirtyFlag() { return m_dirty; }
        Matrix4x4f       GetModelToWorldMatrix();
        Matrix4x4f       GetModelToWorldMatrixLast();
        inline Vector3f  GetScaleLast() { return m_lastFrame.m_scale; }
        inline void      SetGPUResource(Ref<GPUUniformBuffer> buffer, Ref<GPUUniformBuffer> last, Ref<GPUBindId>& bindlessRef,
                 Ref<GPUBindId>& bindlessRefLast)
        {
            m_gpuBuffer          = buffer;
            m_gpuBufferLast      = last;
            m_gpuBindlessRef     = bindlessRef;
            m_gpuBindlessRefLast = bindlessRefLast;
        }
        inline void GetGPUResource(Ref<GPUUniformBuffer>& buffer, Ref<GPUUniformBuffer>& last, Ref<GPUBindId>& bindlessRef,
            Ref<GPUBindId>& bindlessRefLast)
        {
            buffer          = m_gpuBuffer;
            last            = m_gpuBufferLast;
            bindlessRef     = m_gpuBindlessRef;
            bindlessRefLast = m_gpuBindlessRefLast;
        }
        IFRIT_COMPONENT_SERIALIZE(m_attributes);
    };

} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Transform);