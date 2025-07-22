

#include "ifrit/runtime/base/Transform.h"
#include "ifrit/core/math/linalg/LinalgOps.h"
using namespace Ifrit::Math;
namespace Ifrit::Runtime
{

    // Transform

    IFRIT_APIDECL void Transform::SetupProperties()
    {
        AddEnumProperty<TransformUpdateDevice>("Device", m_attributes.m_UpdateDevice,
            { TransformUpdateDevice::CPU, TransformUpdateDevice::GPU }, [&]() { return false; });
        AddProperty<Vector3f, EPropertyEditorType::Text>("Position", m_attributes.m_Position,
            [&]() { return m_attributes.m_UpdateDevice == TransformUpdateDevice::CPU; });
        AddProperty<Vector3f, EPropertyEditorType::Text>("Rotation", m_attributes.m_Rotation,
            [&]() { return m_attributes.m_UpdateDevice == TransformUpdateDevice::CPU; });
        AddProperty<Vector3f, EPropertyEditorType::Text>(
            "Scale", m_attributes.m_Scale, [&]() { return m_attributes.m_UpdateDevice == TransformUpdateDevice::CPU; });
    }

    IFRIT_APIDECL Matrix4x4f Transform::GetModelToWorldMatrix() const
    {
        Matrix4x4f model = Identity<f32, 4>();
        model            = MatMul(Scale3D(m_attributes.m_Scale), model);
        model            = MatMul(EulerAngleToMatrix(m_attributes.m_Rotation), model);
        model            = MatMul(Translate3D(m_attributes.m_Position), model);
        return model;
    }

    IFRIT_APIDECL Matrix4x4f Transform::GetModelToWorldMatrixLast() const
    {
        Matrix4x4f model = Identity<f32, 4>();
        model            = MatMul(Scale3D(m_lastFrame.m_Scale), model);
        model            = MatMul(EulerAngleToMatrix(m_lastFrame.m_Rotation), model);
        model            = MatMul(Translate3D(m_lastFrame.m_Position), model);
        return model;
    }

    IFRIT_APIDECL void Transform::OnFrameCollecting()
    {
        if (m_dirty.changed)
        {
            m_lastFrame = m_attributes;
        }
        m_dirty.lastChanged = m_dirty.changed;
        m_dirty.changed     = false;
    }

    IFRIT_APIDECL void Transform::SetPosition(const Vector3f& pos)
    {
        m_attributes.m_Position = pos;
        m_dirty.changed         = true;
    }

    IFRIT_APIDECL void Transform::SetRotation(const Vector3f& rot)
    {
        m_attributes.m_Rotation = rot;
        m_dirty.changed         = true;
    }

    IFRIT_APIDECL void Transform::SetScale(const Vector3f& scale)
    {
        m_attributes.m_Scale = scale;
        m_dirty.changed      = true;
    }

    IFRIT_APIDECL void Transform::SetDevice(TransformUpdateDevice device)
    {
        IF_LOG_ASSERTION(
            "Transform", m_DeviceOnlyBuffer == nullptr, "Cannot change device mode after GPU resource has been set");
        IF_LOG_ASSERTION(
            "Transform", m_gpuBuffer == nullptr, "Cannot change device mode after GPU resource has been set");
        m_attributes.m_UpdateDevice = device;
        m_dirty.changed             = true;
    }

    IFRIT_APIDECL u32 Transform::GetActiveResourceId()
    {
        if (m_gpuBindlessRef != nullptr)
        {
            return m_gpuBindlessRef->GetActiveId();
        }
        std::abort();
        return 0;
    }

    IFRIT_APIDECL void Transform::SetGPUResource(Ref<GPUUniformBuffer> buffer, Ref<GPUUniformBuffer> last,
        Ref<GPUBindId>& bindlessRef, Ref<GPUBindId>& bindlessRefLast)
    {
        IF_LOG_ASSERTION(
            "Transform", m_attributes.m_UpdateDevice != TransformUpdateDevice::GPU, "Transform not in CPU update mode");
        m_gpuBuffer          = buffer;
        m_gpuBufferLast      = last;
        m_gpuBindlessRef     = bindlessRef;
        m_gpuBindlessRefLast = bindlessRefLast;
    }

    IFRIT_APIDECL void Transform::GetGPUResource(Ref<GPUUniformBuffer>& buffer, Ref<GPUUniformBuffer>& last,
        Ref<GPUBindId>& bindlessRef, Ref<GPUBindId>& bindlessRefLast)
    {
        IF_LOG_ASSERTION(
            "Transform", m_attributes.m_UpdateDevice != TransformUpdateDevice::GPU, "Transform not in CPU update mode");
        buffer          = m_gpuBuffer;
        last            = m_gpuBufferLast;
        bindlessRef     = m_gpuBindlessRef;
        bindlessRefLast = m_gpuBindlessRefLast;
    }

    IFRIT_APIDECL void Transform::GetGPUResourceDeviceMode(RHI::RhiBuffer*& deviceOnlyBuffer)
    {
        IF_LOG_ASSERTION(
            "Transform", m_attributes.m_UpdateDevice == TransformUpdateDevice::GPU, "Transform not in GPU update mode");
        deviceOnlyBuffer = m_DeviceOnlyBuffer.get();
    }

    IFRIT_APIDECL void Transform::SetGPUResourceDeviceMode(RHI::RhiBufferRef deviceOnlyBuffer)
    {
        IF_LOG_ASSERTION(
            "Transform", m_attributes.m_UpdateDevice == TransformUpdateDevice::GPU, "Transform not in GPU update mode");
        m_DeviceOnlyBuffer = deviceOnlyBuffer;
    }

    IFRIT_APIDECL void   Transform::MarkUnchanged() { m_dirty.changed = false; }

} // namespace Ifrit::Runtime
