#include "ifrit/runtime/physics/artemis/rigid/GpuRigidCollider.h"
#include "ifrit/core/math/physics/InertiaTensor.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Runtime::Artemis
{

    IFRIT_APIDECL void GPURigidCollider::SetupProperties()
    {
        AddEnumProperty<GPURigidColliderType>(
            "Collider Type", mType, { GPURigidColliderType::Sphere, GPURigidColliderType::Box });
        AddProperty<Vector3f, EPropertyEditorType::Text>("Cuboid Size", mCuboidSize);
        AddProperty<f32, EPropertyEditorType::Text>("Radius", mRadius);
        AddProperty<f32, EPropertyEditorType::Text>("Mass", mRigidMass);
    }
    IFRIT_APIDECL f32 GPURigidCollider::GetRadius() const { return mRadius; }
    IFRIT_APIDECL f32 GPURigidCollider::GetRigidMass() const { return mRigidMass; }
    IFRIT_APIDECL Shader::Artemis::ERigidColliderType GPURigidCollider::GetColliderType() const
    {
        return static_cast<Shader::Artemis::ERigidColliderType>(mType);
    }
    IFRIT_APIDECL void GPURigidCollider::SetRadius(f32 radius)
    {
        mRadius = radius;
        m_IsDirty             = true;
    }
    IFRIT_APIDECL void     GPURigidCollider::SetColliderType(GPURigidColliderType type) { mType = type; }

    IFRIT_APIDECL void     GPURigidCollider::OnFrameCollecting() { m_IsDirty = false; }
    IFRIT_APIDECL bool     GPURigidCollider::GetIsDirty() const { return m_IsDirty; }
    IFRIT_APIDECL Vector3f GPURigidCollider::GetCuboidSize() const { return mCuboidSize; }

    IFRIT_APIDECL f32      GPURigidCollider::GetMomentOfInertia2D() const
    {
        switch (mType)
        {
            case GPURigidColliderType::Sphere:
                return Math::DiskMomentOfInertiaWrtCenter2D(mRadius, mRigidMass);
            case GPURigidColliderType::Box:
                return Math::RectMomentOfInertiaWrtCenter2D(
                    mCuboidSize.x, mCuboidSize.y, mRigidMass);
            default:
                IF_LOG_ERROR("GPURigidCollider", "Unsupported collider type for moment of inertia calculation");
                return 0.0f;
        }
    }

    IFRIT_APIDECL void GPURigidCollider::SetInternalRigidId(u32 id) { m_InternalRigidId = id; }
    IFRIT_APIDECL u32  GPURigidCollider::GetInternalRigidId() { return m_InternalRigidId; }

} // namespace Ifrit::Runtime::Artemis
