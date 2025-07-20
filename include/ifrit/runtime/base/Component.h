
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
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Property.h"
#include "ifrit/core/typing/EnumReflection.h"
#include "ifrit/core/typing/Traits.h"
#include <typeinfo>

#define IFRIT_COMPONENT_SERIALIZE(...) IFRIT_STRUCT_SERIALIZE(m_id, m_parentObject, __VA_ARGS__)
#define IFRIT_COMPONENT_REGISTER(x) \
    IFRIT_DERIVED_REGISTER(x);      \
    IFRIT_INHERIT_REGISTER(Ifrit::Runtime::Component, x);

// Change log at 2025-07-18:
// It's the pity that the messy ownership identification is found by @AEMShana
// I am changing the ownership of Component to be unique.

namespace Ifrit::Runtime
{

    struct ComponentIdentifier
    {
        GUID   m_GUID;
        String m_Name;
        u32    m_ArrayIndex   = 0;
        u32    m_ManagerIndex = 0;

        IFRIT_STRUCT_SERIALIZE(m_GUID, m_Name)
    };

    template <class T> class AttributeOwner
    {
    protected:
        T m_attributes{};

    public:
        inline String SerializeAttribute()
        {
            String serialized;
            Serialization::SerializeBinary(m_attributes, serialized);
            return serialized;
        }
        inline void DeserializeAttribute()
        {
            String serialized;
            Serialization::DeserializeBinary(serialized, m_attributes);
        }
    };

    class Component;
    class GameObject;
    class Transform;
    class ComponentManager;

    using ComponentTypeHash  = u64;
    using ComponentReference = Pair<ComponentTypeHash, u32>;

    class IFRIT_APIDECL IComponentManagerKeeper
    {
    public:
        virtual ComponentManager* GetComponentManager() = 0;
    };

    class IFRIT_APIDECL ComponentManager : public NonCopyable
    {
    private:
        Queue<u32>                                        m_FreeIdQueue;
        HashMap<ComponentTypeHash, Vec<Owner<Component>>> m_ComponentArray;
        HashMap<u32, ComponentTypeHash>                   m_IdToTypeHash;
        u32                                               m_AllocatedComponents = 0;

    public:
        ComponentManager();
        void RequestRemove(Component* component);
        u32  AllocateId();

    private:
        void                          SetComponentId(Component* component, u32 arrayPos, ComponentTypeHash typeHash);
        inline Vec<Owner<Component>>& GetComponentArray(ComponentTypeHash typeHash)
        {
            return m_ComponentArray[typeHash];
        }

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)>
        ComponentReference CreateComponent(Ref<GameObject> parentObject)
        {
            auto typeName = TTypeInfo<T>::Name;
            auto typeHash = TTypeInfo<T>::Hash;
            if (m_ComponentArray.count(typeHash) == 0)
            {
                m_ComponentArray[typeHash] = Vec<Owner<Component>>();
            }
            auto ret = MakeOwner<T>(parentObject);
            SetComponentId(ret.get(), SizeCast<u32>(m_ComponentArray[typeHash].size()), typeHash);
            m_ComponentArray[typeHash].push_back(std::move(ret));
            return { typeHash, SizeCast<u32>(m_ComponentArray[typeHash].size() - 1) };
        }

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)>
        T* GetComponentFromReference(ComponentReference ref)
        {
            auto typeHash = ref.first;
            if (ref.first != typeHash || m_ComponentArray.count(typeHash) == 0)
                return nullptr;
            auto& componentArray = m_ComponentArray[typeHash];
            if (ref.second >= componentArray.size())
                return nullptr;
            return static_cast<T*>(componentArray[ref.second].get());
        }

        friend class GameObject;
    };

    class IFRIT_APIDECL GameObjectManager : public NonCopyable
    {
    private:
        Vec<Owner<GameObject>> m_GameObjects;
        HashMap<String, u32>   m_GameObjectNameToIndex;
        HashMap<String, u32>   m_GameObjectUUIDToIndex;
    };

    // TODO: for performance considerations, components container is not consistent
    // across different build envs.

    class IFRIT_APIDECL GameObject : public NonCopyable, public std::enable_shared_from_this<GameObject>
    {
    protected:
        ComponentIdentifier             m_Identifier;
        HashMap<ComponentTypeHash, u32> m_ComponentsHashed;
        ComponentManager*               m_ComponentManager = nullptr;

    public:
        GameObject();
        virtual ~GameObject();
        void                   Initialize(ComponentManager* manager);
        inline String          GetName() const { return m_Identifier.m_Name; }
        inline GUID            GetUUID() const { return m_Identifier.m_GUID; }

        // DEPRECATING
        static Ref<GameObject> CreatePrefab(IComponentManagerKeeper* managerKeeper);

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)> T* AddComponent()
        {
            auto componentRef = m_ComponentManager->CreateComponent<T>(shared_from_this());
            auto typeName     = TTypeInfo<T>::Name;
            auto typeHash     = TTypeInfo<T>::Hash;
            if (m_ComponentsHashed.count(typeHash) > 0)
            {
                IF_LOG_ERROR("Component", "Component type name conflicted");
                std::abort();
            }
            m_ComponentsHashed[typeHash] = componentRef.second;
            return m_ComponentManager->GetComponentFromReference<T>(componentRef);
        }

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)> T* GetComponent()
        {
            auto typeHash = TTypeInfo<T>::Hash;
            if (m_ComponentsHashed.count(typeHash) == 0)
            {
                return nullptr;
            }
            auto itIndex   = m_ComponentsHashed[typeHash];
            auto component = m_ComponentManager->GetComponentFromReference<T>({ typeHash, itIndex });
            return component ? component : nullptr;
        }

        inline Vec<Component*> GetAllComponents()
        {
            Vec<Component*> components;
            for (auto& [typeHash, index] : m_ComponentsHashed)
            {
                auto component = m_ComponentManager->GetComponentFromReference<Component>({ typeHash, index });
                if (component)
                    components.push_back(component);
            }
            return components;
        }

        inline void SetName(const String& name) { m_Identifier.m_Name = name; }
        IFRIT_STRUCT_SERIALIZE(m_Identifier, m_ComponentsHashed);
    };

    class IFRIT_APIDECL Component : public NonCopyable
    {
    protected:
        ComponentIdentifier        m_id;
        std::weak_ptr<GameObject>  m_parentObject;
        Vec<ComponentPropertyBase> m_Property;
        bool                       m_PropertyRegistered = false;

        bool                       m_isEnabled         = true;
        bool                       m_shouldInvokeStart = true;
        bool                       m_shouldInvokeAwake = true;

    private:
        GameObject*                m_parentObjectRaw = nullptr;
        inline ComponentIdentifier GetMetaData() { return m_id; }
        friend class ComponentManager;

    protected:
        template <typename T, EPropertyEditorType E, typename... Args>
        inline void AddProperty(const char* name, T& value, Args&&... args)
        {
            m_Property.push_back(
                ComponentProperty<T, E>(name, value, PropertyConstraint<E, T>(std::forward<Args>(args)...)));
        }

        template <typename T,
            typename U = std::underlying_type<T>::type IF_REQUIRES(std::is_enum_v<T>&& TypeIsAnyOf_v<U, i32, u8, i8>)>
        inline void AddEnumProperty(const char* name, T& value, const Vec<T>& enumValues)
        {
            using TUnderlying = typename std::underlying_type<T>::type;
            static_assert(std::is_same_v<TUnderlying, U>, "Invalid enum type");

            Vec<Pair<U, String>> enumOptions;
            for (auto& enumValue : enumValues)
            {
                enumOptions.push_back({ GetEnumUnderlyingValue(enumValue), GetEnumName(enumValue) });
            }
            m_Property.push_back(ComponentProperty<U, EPropertyEditorType::Select>(
                name, reinterpret_cast<U&>(value), std::move(enumOptions)));
        }

        inline virtual void AddProperty(ComponentPropertyBase prop) final { m_Property.push_back(prop); }

    public:
        virtual void CallPropertyEditorHandle();

    public:
        Component() { IntializeComponent(); }; // for deserializatioin
        Component(Ref<GameObject> parentObject);
        virtual ~Component() = default;

        virtual String               Serialize()   = 0;
        virtual void                 Deserialize() = 0;
        virtual void                 IntializeComponent();

        virtual void                 OnFrameCollecting() {}
        virtual void                 OnAwake() {}
        virtual void                 OnStart() {}
        virtual void                 OnFixedUpdate() {}
        virtual void                 OnUpdate() {}
        virtual void                 OnEnd() {}

        virtual void                 SetupProperties() = 0;
        inline u32                   GetNumProperties() const { return SizeCast<u32>(m_Property.size()) + 1; }

        inline void                  SetName(const String& name) { m_id.m_Name = name; }
        virtual void                 SetAssetReferencedAttributes(const Vec<Ref<IAssetCompatible>>& out) {}
        void                         SetEnable(bool enable);

        inline String                GetName() const { return m_id.m_Name; }
        inline GUID                  GetGUID() const { return m_id.m_GUID; }
        inline Ref<GameObject>       GetParent() const { return m_parentObject.lock(); }
        virtual Vec<AssetReference*> GetAssetRefs() { return {}; }
        inline bool                  IsEnabled() const { return m_isEnabled; }

        void                         InvokeStart();
        void                         InvokeAwake();

        // This function is intended to be used in performance-critical code.
        // Use with caution.
        inline GameObject*           GetParentUnsafe() { return m_parentObjectRaw; }

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
        using GPUUniformBuffer                     = Ifrit::RHI::RhiMultiBuffer;
        using GPUBindId                            = Ifrit::RHI::RhiDescHandleLegacy;
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
        Transform(Ref<GameObject> parent) : Component(parent), AttributeOwner<TransformAttribute>() {}

        String      Serialize() override { return SerializeAttribute(); }
        void        Deserialize() override { DeserializeAttribute(); }

        void        SetupProperties() override;

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
        inline void      SetGPUResource(Ref<GPUUniformBuffer> buffer, Ref<GPUUniformBuffer> last,
                 Ref<GPUBindId>& bindlessRef, Ref<GPUBindId>& bindlessRefLast)
        {
            m_gpuBuffer          = buffer;
            m_gpuBufferLast      = last;
            m_gpuBindlessRef     = bindlessRef;
            m_gpuBindlessRefLast = bindlessRefLast;
        }
        inline void GetGPUResource(Ref<GPUUniformBuffer>& buffer, Ref<GPUUniformBuffer>& last,
            Ref<GPUBindId>& bindlessRef, Ref<GPUBindId>& bindlessRefLast)
        {
            buffer          = m_gpuBuffer;
            last            = m_gpuBufferLast;
            bindlessRef     = m_gpuBindlessRef;
            bindlessRefLast = m_gpuBindlessRefLast;
        }
        inline u32 GetActiveResourceId()
        {
            if (m_gpuBindlessRef != nullptr)
            {
                return m_gpuBindlessRef->GetActiveId();
            }
            std::abort();
            return 0;
        }
        IFRIT_COMPONENT_SERIALIZE(m_attributes);
    };

} // namespace Ifrit::Runtime

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Transform);
