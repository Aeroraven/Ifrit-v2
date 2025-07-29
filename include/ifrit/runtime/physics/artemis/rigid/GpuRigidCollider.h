#pragma once

#include "ifrit/runtime/base/Component.h"
#include "ifrit/core/serialization/SerialEnumDefine.h"
#include "ifrit/core/serialization/MathTypeSerialization.h"
#include "ifrit.shader.neo/Shared/Artemis/Rigid.Shared.h"

namespace Ifrit::Runtime::Artemis
{

    enum class GPURigidColliderType : u8
    {
        Sphere = 0,
        Box    = 1
    };

    class IFRIT_RUNTIME_API IF_CLASS() GPURigidCollider : public Component
    {
    public:
        IF_PROPERTY(Editable, UIText)
        Vector3f mCuboidSize = Vector3f(0.1f, 0.1f, 0.1f);

        IF_PROPERTY(Editable, UIText)
        f32 mRadius = 1.0f;

        IF_PROPERTY(Editable, UIText)
        f32 mRigidMass = 1.14514f;

        IF_PROPERTY(Editable, UISelect)
        GPURigidColliderType mType = GPURigidColliderType::Sphere;

    private:
        u32  m_InternalRigidId = ~0u;
        bool m_IsDirty         = true;

    public:
        GPURigidCollider() {};
        GPURigidCollider(GameObject* parent) : Component(parent) {}

        f32                                 GetRadius() const;
        f32                                 GetRigidMass() const;
        Shader::Artemis::ERigidColliderType GetColliderType() const;
        bool                                GetIsDirty() const;
        Vector3f                            GetCuboidSize() const;
        f32                                 GetMomentOfInertia2D() const;

        void                                SetRadius(f32 radius);
        void                                SetColliderType(GPURigidColliderType type);
        void                                OnFrameCollecting() override;

        // TODO: reconsider access control
        void                                SetInternalRigidId(u32 id);
        u32                                 GetInternalRigidId();

        IFRIT_COMPONENT_SERIALIZE(m_attributes);
    };

} // namespace Ifrit::Runtime::Artemis
IFRIT_ENUMCLASS_SERIALIZE(Ifrit::Runtime::Artemis::GPURigidColliderType);

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Artemis::GPURigidCollider)
