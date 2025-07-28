#pragma once
#include "ifrit/runtime/base/Component.h"
#include "ifrit/core/serialization/SerialEnumDefine.h"
#include "ifrit/core/reflection/ReflAttrs.h"

namespace Ifrit::Runtime
{
    enum class TransformUpdateDevice : u8
    {
        CPU,
        GPU,
    };

    struct TransformAttributeLast
    {
        Vector3f m_Position;
        Vector3f m_Rotation;
        Vector3f m_Scale;
    };

    class IFRIT_APIDECL IF_CLASS() Transform : public Component
    {
    public:
        IF_PROPERTY()
        TransformUpdateDevice mUpdateDevice = TransformUpdateDevice::CPU;

        IF_PROPERTY()
        Vector3f mPosition = Vector3f{ 0.0f, 0.0f, 0.0f };

        IF_PROPERTY()
        Vector3f mRotation = Vector3f{ 0.0f, 0.0f, 0.0f };

        IF_PROPERTY()
        Vector3f mScale = Vector3f{ 1.0f, 1.0f, 1.0f };

    private:
        using GPUUniformBuffer = Ifrit::RHI::RhiMultiBuffer;
        using GPUBindId        = Ifrit::RHI::RhiDescHandleLegacy;

        RHI::RhiBufferRef      m_DeviceOnlyBuffer = nullptr;

        Ref<GPUUniformBuffer>  m_gpuBuffer          = nullptr;
        Ref<GPUUniformBuffer>  m_gpuBufferLast      = nullptr;
        Ref<GPUBindId>         m_gpuBindlessRef     = nullptr;
        Ref<GPUBindId>         m_gpuBindlessRefLast = nullptr;
        TransformAttributeLast m_lastFrame;

        struct DirtyFlag
        {
            bool changed     = true;
            bool lastChanged = true;
        } m_dirty;

    public:
        Transform() {};
        Transform(GameObject* parent) : Component(parent) {}

        void                         SetupProperties() override;
        void                         OnFrameCollecting();

        // getters
        inline Vector3f              GetPosition() const { return mPosition; }
        inline Vector3f              GetRotation() const { return mRotation; }
        inline Vector3f              GetScale() const { return mScale; }
        inline Vector3f              GetScaleLast() const { return mScale; }
        inline DirtyFlag             GetDirtyFlag() { return m_dirty; }
        inline TransformUpdateDevice GetUpdateDevice() const { return mUpdateDevice; }

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
