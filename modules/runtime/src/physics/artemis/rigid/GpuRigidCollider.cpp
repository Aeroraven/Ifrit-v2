#include "ifrit/runtime/physics/artemis/rigid/GpuRigidCollider.h"

namespace Ifrit::Runtime::Artemis
{
    IFRIT_APIDECL String GPURigidCollider::Serialize() { return SerializeAttribute(); }
    IFRIT_APIDECL void   GPURigidCollider::Deserialize() { DeserializeAttribute(); }
    IFRIT_APIDECL void   GPURigidCollider::SetupProperties()
    {
        AddProperty<f32, EPropertyEditorType::Text>("Radius", m_attributes.m_Radius);
    }
    IFRIT_APIDECL f32  GPURigidCollider::GetRadius() const { return m_attributes.m_Radius; }
    IFRIT_APIDECL void GPURigidCollider::SetRadius(f32 radius)
    {
        m_attributes.m_Radius = radius;
        m_IsDirty             = true;
    }

    IFRIT_APIDECL void GPURigidCollider::OnFrameCollecting() { m_IsDirty = false; }
    IFRIT_APIDECL bool GPURigidCollider::GetIsDirty() const { return m_IsDirty; }

    IFRIT_APIDECL void GPURigidCollider::SetInternalRigidId(u32 id) { m_InternalRigidId = id; }
    IFRIT_APIDECL u32  GPURigidCollider::GetInternalRigidId() { return m_InternalRigidId; }

} // namespace Ifrit::Runtime::Artemis
