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
        bool m_IsDirty = true;

    public:
        String Serialize() override;
        void   Deserialize() override;
        void   SetupProperties() override;

        f32    GetRadius() const;
        bool   GetIsDirty() const;

        void   SetRadius(f32 radius);
        void   OnFrameCollecting() override;

        IFRIT_COMPONENT_SERIALIZE(m_attributes);
    };

} // namespace Ifrit::Runtime::Artemis

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Artemis::GPURigidCollider)