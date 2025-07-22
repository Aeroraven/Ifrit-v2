#pragma once

#include "ifrit/runtime/base/Component.h"

namespace Ifrit::Runtime::Artemis
{

    struct GPURigidColliderProperty
    {
        // SPHERE FOR SIMPLICITY
        f32 m_Radius = 1.0f;

        IFRIT_STRUCT_SERIALIZE(m_Radius);
    };

    class IFRIT_RUNTIME_API GPURigidCollider : public Component, public AttributeOwner<GPURigidColliderProperty>
    {
    private:
        bool m_IsDirty         = true;
        u32  m_InternalRigidId = ~0u;

    public:
        GPURigidCollider() {};
        GPURigidCollider(GameObject* parent) : Component(parent), AttributeOwner<GPURigidColliderProperty>() {}


        void   SetupProperties() override;

        f32    GetRadius() const;
        bool   GetIsDirty() const;

        void   SetRadius(f32 radius);
        void   OnFrameCollecting() override;

        // TODO: reconsider access control
        void   SetInternalRigidId(u32 id);
        u32    GetInternalRigidId();

        IFRIT_COMPONENT_SERIALIZE(m_attributes);
    };

} // namespace Ifrit::Runtime::Artemis

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Artemis::GPURigidCollider)
