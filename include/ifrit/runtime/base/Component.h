#pragma once
#include "AssetReference.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Property.h"
#include "ifrit/core/typing/EnumReflection.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/typing/TypeMetaInfo.h"
#include <typeinfo>

#define IFRIT_COMPONENT_SERIALIZE(...) IFRIT_STRUCT_SERIALIZE(m_id, m_isEnabled, m_ParentRef, __VA_ARGS__)
#define IFRIT_COMPONENT_SERIALIZE_EMPTY() IFRIT_STRUCT_SERIALIZE(m_id, m_isEnabled, m_ParentRef)
#define IFRIT_COMPONENT_REGISTER(x) \
    IFRIT_DERIVED_REGISTER(x);      \
    IFRIT_INHERIT_REGISTER(Ifrit::Runtime::Component, x);

// Change log at 2025-07-18:
// It's the pity that the messy ownership identification is found by @AEMShana
// I am changing the ownership of Component to be unique.

// Change log at 2025-07-23:
// Temporarily droping (sealing) the manual serialization for Component. It's not user friendly to use

namespace Ifrit::Runtime
{

    struct ComponentIdentifier
    {
        GUID   m_GUID;
        String m_Name;
        u32    m_ArrayIndex   = 0;
        u32    m_ManagerIndex = ~0u;

        IFRIT_STRUCT_SERIALIZE(m_GUID, m_Name, m_ArrayIndex, m_ManagerIndex);
    };

    template <class T> class AttributeOwner
    {
    protected:
        T m_attributes{};
    };

    class Component;
    class GameObject;
    class Transform;
    class ComponentManager;
    class GameObjectManager;

    using ComponentTypeHash   = u64;
    using ComponentReference  = Pair<ComponentTypeHash, u32>;
    using GameObjectReference = u32;

    class IFRIT_APIDECL IComponentManagerKeeper
    {
    public:
        virtual ComponentManager*  GetComponentManager()  = 0;
        virtual GameObjectManager* GetGameObjectManager() = 0;
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
        ComponentReference CreateComponent(GameObject* parentObject)
        {
            auto typeName = TMetaTypeInfo<T>::Name;
            auto typeHash = TMetaTypeInfo<T>::Hash;
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

    public:
        IFRIT_STRUCT_SERIALIZE(m_FreeIdQueue, m_ComponentArray, m_IdToTypeHash, m_AllocatedComponents);
    };

    class IFRIT_APIDECL GameObjectManager : public NonCopyable
    {
    private:
        Queue<u32>             m_FreeIdQueue;
        Vec<Owner<GameObject>> m_GameObjects;
        HashMap<String, u32>   m_GameObjectNameToIndex;
        HashMap<GUID, u32>     m_GameObjectUUIDToIndex;
        u32                    m_AllocatedObjects = 0;

    private:
        u32 AllocateId();

    public:
        GameObjectManager()  = default;
        ~GameObjectManager() = default;

        GameObjectReference CreateGameObject(const String& name);
        GameObject*         GetGameObject(GameObjectReference ref);

        void                RequestRemove(GameObjectReference ref);

        IFRIT_STRUCT_SERIALIZE(
            m_FreeIdQueue, m_GameObjects, m_GameObjectNameToIndex, m_GameObjectUUIDToIndex, m_AllocatedObjects);
    };

    // TODO: for performance considerations, components container is not consistent
    // across different build envs.

    class IFRIT_APIDECL GameObject : public NonCopyable
    {
    protected:
        ComponentIdentifier             m_Identifier;
        HashMap<ComponentTypeHash, u32> m_ComponentsHashed;
        ComponentManager*               m_ComponentManager  = nullptr;
        GameObjectManager*              m_GameObjectManager = nullptr;

    private:
        inline void SetManagerId(GameObjectReference id) { m_Identifier.m_ManagerIndex = id; }

    public:
        GameObject();
        virtual ~GameObject();
        void                       Initialize(ComponentManager* manager, GameObjectManager* gameObjectManager);
        inline String              GetName() const { return m_Identifier.m_Name; }
        inline GUID                GetUUID() const { return m_Identifier.m_GUID; }
        inline GameObjectReference GetManagerId() const { return m_Identifier.m_ManagerIndex; }

        // DEPRECATING
        static GameObject*         CreatePrefab(IComponentManagerKeeper* managerKeeper);

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)> T* AddComponent()
        {
            auto componentRef = m_ComponentManager->CreateComponent<T>(this);
            auto typeName     = TMetaTypeInfo<T>::Name;
            auto typeHash     = TMetaTypeInfo<T>::Hash;
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
            auto typeHash = TMetaTypeInfo<T>::Hash;
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

        friend class GameObjectManager;
        friend class Component;

        inline void SetName(const String& name) { m_Identifier.m_Name = name; }
        IFRIT_STRUCT_SERIALIZE(m_Identifier, m_ComponentsHashed);
    };

    class IFRIT_APIDECL Component : public NonCopyable
    {
    protected:
        ComponentIdentifier        m_id;
        GameObjectReference        m_ParentRef;
        GameObjectManager*         m_GameObjectManager = nullptr;

        Vec<ComponentPropertyBase> m_Property;
        bool                       m_PropertyRegistered = false;

        bool                       m_isEnabled         = true;
        bool                       m_shouldInvokeStart = true;
        bool                       m_shouldInvokeAwake = true;

    private:
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
        inline void AddEnumProperty(
            const char* name, T& value, const Vec<T>& enumValues, Fn<bool()> predicate = PropertyPredicateAlwaysTrue)
        {
            using TUnderlying = typename std::underlying_type<T>::type;
            static_assert(std::is_same_v<TUnderlying, U>, "Invalid enum type");

            Vec<Pair<U, String>> enumOptions;
            for (auto& enumValue : enumValues)
            {
                enumOptions.push_back({ GetEnumUnderlyingValue(enumValue), GetEnumName(enumValue) });
            }
            m_Property.push_back(ComponentProperty<U, EPropertyEditorType::Select>(name, reinterpret_cast<U&>(value),
                PropertyConstraint<EPropertyEditorType::Select, U>(enumOptions, predicate)));
        }

        inline virtual void AddProperty(ComponentPropertyBase prop) final { m_Property.push_back(prop); }

    public:
        virtual void CallPropertyEditorHandle();

    public:
        Component() {}; // for deserializatioin
        Component(GameObject* parentObject);
        virtual ~Component() = default;

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
        GameObject*                  GetParent() const;
        virtual Vec<AssetReference*> GetAssetRefs() { return {}; }
        inline bool                  IsEnabled() const { return m_isEnabled; }

        void                         InvokeStart();
        void                         InvokeAwake();

        IFRIT_STRUCT_SERIALIZE(m_id);
    };

} // namespace Ifrit::Runtime
