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

    struct GPURigidColliderProperty
    {
        // SPHERE FOR SIMPLICITY
        Vector3f             m_CuboidSize = Vector3f(0.1f, 0.1f, 0.1f);
        f32                  m_Radius     = 1.0f;
        f32                  m_RigidMass  = 1.14514f;
        GPURigidColliderType m_Type       = GPURigidColliderType::Sphere;

        IFRIT_STRUCT_SERIALIZE(m_CuboidSize, m_Radius, m_RigidMass, m_Type);
    };

    class IFRIT_RUNTIME_API GPURigidCollider : public Component, public AttributeOwner<GPURigidColliderProperty>
    {
    private:
        u32  m_InternalRigidId = ~0u;
        bool m_IsDirty         = true;

    public:
        GPURigidCollider() {};
        GPURigidCollider(GameObject* parent) : Component(parent), AttributeOwner<GPURigidColliderProperty>() {}

        void                                SetupProperties() override;

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
