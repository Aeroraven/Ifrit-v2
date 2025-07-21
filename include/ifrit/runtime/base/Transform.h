#pragma once
#include "ifrit/runtime/base/Component.h"
#include "ifrit/core/serialization/SerialEnumDefine.h"

namespace Ifrit::Runtime
{
    enum class TransformUpdateDevice : u8
    {
        CPU,
        GPU,
    };

    struct TransformAttribute
    {
        TransformUpdateDevice m_UpdateDevice = TransformUpdateDevice::CPU;
        Vector3f              m_Position     = Vector3f{ 0.0f, 0.0f, 0.0f };
        Vector3f              m_Rotation     = Vector3f{ 0.0f, 0.0f, 0.0f };
        Vector3f              m_Scale        = Vector3f{ 1.0f, 1.0f, 1.0f };

        IFRIT_STRUCT_SERIALIZE(m_UpdateDevice, m_Position, m_Rotation, m_Scale);
    };

    class IFRIT_APIDECL Transform : public Component, public AttributeOwner<TransformAttribute>
    {
    private:
        using GPUUniformBuffer = Ifrit::RHI::RhiMultiBuffer;
        using GPUBindId        = Ifrit::RHI::RhiDescHandleLegacy;

        RHI::RhiBufferRef m_DeviceOnlyBuffer = nullptr;

        Ref<GPUUniformBuffer> m_gpuBuffer          = nullptr;
        Ref<GPUUniformBuffer> m_gpuBufferLast      = nullptr;
        Ref<GPUBindId>        m_gpuBindlessRef     = nullptr;
        Ref<GPUBindId>        m_gpuBindlessRefLast = nullptr;
        TransformAttribute    m_lastFrame;

        struct DirtyFlag
        {
            bool changed     = true;
            bool lastChanged = true;
        } m_dirty;

    public:
        Transform() {};
        Transform(GameObject* parent) : Component(parent), AttributeOwner<TransformAttribute>() {}

        String                       Serialize() override;
        void                         Deserialize() override;

        void                         SetupProperties() override;
        void                         OnFrameCollecting();

        // getters
        inline Vector3f              GetPosition() const { return m_attributes.m_Position; }
        inline Vector3f              GetRotation() const { return m_attributes.m_Rotation; }
        inline Vector3f              GetScale() const { return m_attributes.m_Scale; }
        inline Vector3f              GetScaleLast() const { return m_lastFrame.m_Scale; }
        inline DirtyFlag             GetDirtyFlag() { return m_dirty; }
        inline TransformUpdateDevice GetUpdateDevice() const { return m_attributes.m_UpdateDevice; }

        // setters
        void                         SetPosition(const Vector3f& pos);
        void                         SetRotation(const Vector3f& rot);
        void                         SetScale(const Vector3f& scale);
        void                         SetDevice(TransformUpdateDevice device);
        void                         MarkUnchanged();

        Matrix4x4f                   GetModelToWorldMatrix() const;
        Matrix4x4f                   GetModelToWorldMatrixLast() const;

        void SetGPUResource(Ref<GPUUniformBuffer> buffer, Ref<GPUUniformBuffer> last, Ref<GPUBindId>& bindlessRef,
            Ref<GPUBindId>& bindlessRefLast);
        void GetGPUResource(Ref<GPUUniformBuffer>& buffer, Ref<GPUUniformBuffer>& last, Ref<GPUBindId>& bindlessRef,
            Ref<GPUBindId>& bindlessRefLast);

        void GetGPUResourceDeviceMode(RHI::RhiBuffer*& deviceOnlyBuffer);
        void SetGPUResourceDeviceMode(RHI::RhiBufferRef deviceOnlyBuffer);

        u32  GetActiveResourceId();
        IFRIT_COMPONENT_SERIALIZE(m_attributes);
    };
} // namespace Ifrit::Runtime

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Transform);
IFRIT_ENUMCLASS_SERIALIZE(Ifrit::Runtime::TransformUpdateDevice)