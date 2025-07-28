#pragma once
#include "ifrit/runtime/base/AssetReference.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Property.h"
#include "ifrit/core/typing/EnumReflection.h"
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/typing/TypeMetaInfo.h"
#include <typeinfo>
#include "ifrit/core/reflection/ReflAttrs.h"

#define IFRIT_COMPONENT_SERIALIZE(...)    // IFRIT_STRUCT_SERIALIZE(m_id, m_isEnabled, m_ParentRef, __VA_ARGS__)
#define IFRIT_COMPONENT_SERIALIZE_EMPTY() // IFRIT_STRUCT_SERIALIZE(m_id, m_isEnabled, m_ParentRef)
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

    class IFRIT_APIDECL IF_CLASS() ComponentManager : public NonCopyable
    {
    public:
        IF_PROPERTY()
        HashMap<ComponentTypeHash, Vec<Owner<Component>>> mComponentArray;

        IF_PROPERTY()
        HashMap<u32, ComponentTypeHash> mIdToTypeHash;

        IF_PROPERTY()
        u32 mAllocatedComponents = 0;

    private:
        Queue<u32> m_FreeIdQueue;

    public:
        ComponentManager();
        void RequestRemove(Component* component);
        u32  AllocateId();

    private:
        void                          SetComponentId(Component* component, u32 arrayPos, ComponentTypeHash typeHash);
        inline Vec<Owner<Component>>& GetComponentArray(ComponentTypeHash typeHash)
        {
            return mComponentArray[typeHash];
        }

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)>
        ComponentReference CreateComponent(GameObject* parentObject)
        {
            auto typeName = TMetaTypeInfo<T>::Name;
            auto typeHash = TMetaTypeInfo<T>::Hash;
            if (mComponentArray.count(typeHash) == 0)
            {
                mComponentArray[typeHash] = Vec<Owner<Component>>();
            }
            auto ret = MakeOwner<T>(parentObject);
            SetComponentId(ret.get(), SizeCast<u32>(mComponentArray[typeHash].size()), typeHash);
            mComponentArray[typeHash].push_back(std::move(ret));
            return { typeHash, SizeCast<u32>(mComponentArray[typeHash].size() - 1) };
        }

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)>
        T* GetComponentFromReference(ComponentReference ref)
        {
            auto typeHash = ref.first;
            if (ref.first != typeHash || mComponentArray.count(typeHash) == 0)
                return nullptr;
            auto& componentArray = mComponentArray[typeHash];
            if (ref.second >= componentArray.size())
                return nullptr;
            return static_cast<T*>(componentArray[ref.second].get());
        }

        friend class GameObject;
    };

    class IFRIT_APIDECL IF_CLASS() GameObjectManager : public NonCopyable
    {

    public:
        IF_PROPERTY()
        Vec<Owner<GameObject>> mGameObjects;

        IF_PROPERTY()
        u32 mAllocatedObjects = 0;

    private:
        Queue<u32>           m_FreeIdQueue;
        HashMap<String, u32> m_GameObjectNameToIndex;
        HashMap<GUID, u32>   m_GameObjectUUIDToIndex;

    private:
        u32 AllocateId();

    public:
        GameObjectManager()  = default;
        ~GameObjectManager() = default;

        GameObjectReference CreateGameObject(const String& name);
        GameObject*         GetGameObject(GameObjectReference ref);

        void                RequestRemove(GameObjectReference ref);
    };

    // TODO: for performance considerations, components container is not consistent
    // across different build envs.

    class IFRIT_APIDECL IF_CLASS() GameObject : public NonCopyable
    {
    public:
        IF_PROPERTY()
        u32 mManagedIndex;

        IF_PROPERTY()
        u32 mId;

        IF_PROPERTY()
        GUID mGuid;

        IF_PROPERTY()
        String mName;

        IF_PROPERTY()
        HashMap<ComponentTypeHash, u32> mComponentsHashed;

    public:
        ComponentManager*  m_ComponentManager  = nullptr;
        GameObjectManager* m_GameObjectManager = nullptr;

    private:
        inline void SetManagerId(GameObjectReference id) { mManagedIndex = id; }

    public:
        GameObject();
        virtual ~GameObject();
        void                       Initialize(ComponentManager* manager, GameObjectManager* gameObjectManager);
        inline String              GetName() const { return mName; }
        inline GUID                GetUUID() const { return mGuid; }
        inline GameObjectReference GetManagerId() const { return mManagedIndex; }

        // DEPRECATING
        static GameObject*         CreatePrefab(IComponentManagerKeeper* managerKeeper);

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)> T* AddComponent()
        {
            auto componentRef = m_ComponentManager->CreateComponent<T>(this);
            auto typeName     = TMetaTypeInfo<T>::Name;
            auto typeHash     = TMetaTypeInfo<T>::Hash;
            if (mComponentsHashed.count(typeHash) > 0)
            {
                // IF_LOG_ERROR("Component", "Component type name conflicted");
                std::abort();
            }
            mComponentsHashed[typeHash] = componentRef.second;
            return m_ComponentManager->GetComponentFromReference<T>(componentRef);
        }

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)> T* GetComponent()
        {
            auto typeHash = TMetaTypeInfo<T>::Hash;
            if (mComponentsHashed.count(typeHash) == 0)
            {
                return nullptr;
            }
            auto itIndex   = mComponentsHashed[typeHash];
            auto component = m_ComponentManager->GetComponentFromReference<T>({ typeHash, itIndex });
            return component ? component : nullptr;
        }

        inline Vec<Component*> GetAllComponents()
        {
            Vec<Component*> components;
            for (auto& [typeHash, index] : mComponentsHashed)
            {
                auto component = m_ComponentManager->GetComponentFromReference<Component>({ typeHash, index });
                if (component)
                    components.push_back(component);
            }
            return components;
        }

        friend class GameObjectManager;
        friend class Component;

        inline void SetName(const String& name) { mName = name; }
    };

    class IFRIT_APIDECL IF_CLASS() Component : public NonCopyable
    {
    public:
        IF_PROPERTY()
        u32 mId;

        IF_PROPERTY()
        String mName;

        IF_PROPERTY()
        GUID mGuid;

        IF_PROPERTY()
        u32 mManagedIndex;

        IF_PROPERTY()
        u32 mArrayIndex;

        IF_PROPERTY(Editable, UISelect)
        bool mEnabled = true;

    public:
        GameObjectReference        m_ParentRef;
        GameObjectManager*         m_GameObjectManager = nullptr;

        Vec<ComponentPropertyBase> m_Property;
        bool                       m_PropertyRegistered = false;
        bool                       m_shouldInvokeStart  = true;
        bool                       m_shouldInvokeAwake  = true;

    private:
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
        inline u32   GetArrayIndex() const { return mArrayIndex; }
        inline u32   GetManagedIndex() const { return mManagedIndex; }

    public:
        Component(){}; // for deserializatioin
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

        inline void                  SetName(const String& name) { mName = name; }
        virtual void                 SetAssetReferencedAttributes(const Vec<Ref<IAssetCompatible>>& out) {}
        void                         SetEnable(bool enable);

        inline String                GetName() const { return mName; }
        inline GUID                  GetGUID() const { return mGuid; }
        GameObject*                  GetParent() const;
        virtual Vec<AssetReference*> GetAssetRefs() { return {}; }
        inline bool                  IsEnabled() const { return mEnabled; }

        void                         InvokeStart();
        void                         InvokeAwake();
    };

} // namespace Ifrit::Runtime
