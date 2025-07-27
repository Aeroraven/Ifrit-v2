#include "ifrit/runtime/physics/artemis/rigid/GpuRigidCollider.h"
#include "ifrit/core/math/physics/InertiaTensor.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Runtime::Artemis
{

    IFRIT_APIDECL void GPURigidCollider::SetupProperties()
    {
        AddEnumProperty<GPURigidColliderType>(
            "Collider Type", m_attributes.m_Type, { GPURigidColliderType::Sphere, GPURigidColliderType::Box });
        AddProperty<Vector3f, EPropertyEditorType::Text>("Cuboid Size", m_attributes.m_CuboidSize);
        AddProperty<f32, EPropertyEditorType::Text>("Radius", m_attributes.m_Radius);
        AddProperty<f32, EPropertyEditorType::Text>("Mass", m_attributes.m_RigidMass);
    }
    IFRIT_APIDECL f32 GPURigidCollider::GetRadius() const { return m_attributes.m_Radius; }
    IFRIT_APIDECL f32 GPURigidCollider::GetRigidMass() const { return m_attributes.m_RigidMass; }
    IFRIT_APIDECL Shader::Artemis::ERigidColliderType GPURigidCollider::GetColliderType() const
    {
        return static_cast<Shader::Artemis::ERigidColliderType>(m_attributes.m_Type);
    }
    IFRIT_APIDECL void GPURigidCollider::SetRadius(f32 radius)
    {
        m_attributes.m_Radius = radius;
        m_IsDirty             = true;
    }
    IFRIT_APIDECL void     GPURigidCollider::SetColliderType(GPURigidColliderType type) { m_attributes.m_Type = type; }

    IFRIT_APIDECL void     GPURigidCollider::OnFrameCollecting() { m_IsDirty = false; }
    IFRIT_APIDECL bool     GPURigidCollider::GetIsDirty() const { return m_IsDirty; }
    IFRIT_APIDECL Vector3f GPURigidCollider::GetCuboidSize() const { return m_attributes.m_CuboidSize; }

    IFRIT_APIDECL f32      GPURigidCollider::GetMomentOfInertia2D() const
    {
        switch (m_attributes.m_Type)
        {
            case GPURigidColliderType::Sphere:
                return Math::DiskMomentOfInertiaWrtCenter2D(m_attributes.m_Radius, m_attributes.m_RigidMass);
            case GPURigidColliderType::Box:
                return Math::RectMomentOfInertiaWrtCenter2D(
                    m_attributes.m_CuboidSize.x, m_attributes.m_CuboidSize.y, m_attributes.m_RigidMass);
            default:
                IF_LOG_ERROR("GPURigidCollider", "Unsupported collider type for moment of inertia calculation");
                return 0.0f;
        }
    }

    IFRIT_APIDECL void GPURigidCollider::SetInternalRigidId(u32 id) { m_InternalRigidId = id; }
    IFRIT_APIDECL u32  GPURigidCollider::GetInternalRigidId() { return m_InternalRigidId; }

} // namespace Ifrit::Runtime::Artemis
