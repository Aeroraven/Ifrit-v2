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

        ComponentReference CreateComponentFromMeta(
            GameObject* parentObject, const FMetaTypeInfo& metaTypeInfo, bool enabled);

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

        template <typename T IF_REQUIRES(std::is_base_of<Component, T>::value)> T* AddComponent()
        {
            auto componentRef = m_ComponentManager->CreateComponent<T>(this);
            auto typeName     = TMetaTypeInfo<T>::Name;
            auto typeHash     = TMetaTypeInfo<T>::Hash;
            if (mComponentsHashed.count(typeHash) > 0)
            {
                std::abort();
            }
            mComponentsHashed[typeHash] = componentRef.second;
            return m_ComponentManager->GetComponentFromReference<T>(componentRef);
        }

        void AddComponentFromeMeta(const FMetaTypeInfo& metaTypeInfo, bool enabled);

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
        GameObjectReference m_ParentRef;
        GameObjectManager*  m_GameObjectManager = nullptr;

        // Vec<ComponentPropertyBase> m_Property;
        bool                m_PropertyRegistered = false;
        bool                m_shouldInvokeStart  = true;
        bool                m_shouldInvokeAwake  = true;

    private:
        friend class ComponentManager;

    public:
        virtual void CallPropertyEditorHandle();
        virtual void CallFunctionEditorHandle();

        inline u32   GetArrayIndex() const { return mArrayIndex; }
        inline u32   GetManagedIndex() const { return mManagedIndex; }

    public:
        Component() {}; // for deserializatioin
        Component(GameObject* parentObject);
        virtual ~Component() = default;

        virtual u32   GetNumVisibleProperties() const final;

        virtual void  OnFrameCollecting() {}
        virtual void  OnAwake() {}
        virtual void  OnStart() {}
        virtual void  OnFixedUpdate() {}
        virtual void  OnUpdate() {}
        virtual void  OnEnd() {}

        virtual void  SetupProperties() final {}

        inline void   SetName(const String& name) { mName = name; }
        void          SetEnable(bool enable);

        inline String GetName() const { return mName; }
        inline GUID   GetGUID() const { return mGuid; }
        GameObject*   GetParent() const;
        inline bool   IsEnabled() const { return mEnabled; }

        void          InvokeStart();
        void          InvokeAwake();
    };

} // namespace Ifrit::Runtime
