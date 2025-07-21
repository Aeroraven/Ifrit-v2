#include "ifrit/runtime/scene/FrameComponentUpdate.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/runtime/base/MeshTransform.h"
#include "ifrit/core/math/linalg/LinalgOps.h"

using namespace Ifrit::Math;
using namespace Ifrit::RHI;

namespace Ifrit::Runtime
{

    void FillingMeshTransformData(const Transform* transform, MeshInstanceTransform& model, bool lastFrame = false)
    {
        // Transpose is required because glsl uses column major matrices
        model.model    = Math::Transpose(transform->GetModelToWorldMatrix());
        model.invModel = Math::Transpose(Math::Inverse(transform->GetModelToWorldMatrix()));
        if (lastFrame)
            model.maxScale = Vector4f(transform->GetScaleLast(), 0.0);
        else
            model.maxScale = Vector4f(transform->GetScale(), 0.0);
        model.m_Position = Vector4f(transform->GetPosition(), 1.0);
        model.m_Rotation = Vector4f(transform->GetRotation(), 0.0);
    }

    IFRIT_APIDECL TransformAllocData UpdateTransformGPUData(Transform* transform, RHI::RhiBackend* rhi)
    {
        TransformAllocData ret;
        ret.m_Changed          = false;
        ret.m_TransformRef     = ~0u;
        ret.m_TransformRefLast = ~0u;

        auto updateDevice = transform->GetUpdateDevice();

        if (updateDevice == TransformUpdateDevice::CPU)
        {

            Ref<RhiMultiBuffer>      transformBuffer     = nullptr;
            Ref<RhiMultiBuffer>      transformBufferLast = nullptr;
            Ref<RhiDescHandleLegacy> bindlessRef         = nullptr;
            Ref<RhiDescHandleLegacy> bindlessRefLast     = nullptr;
            transform->GetGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
            bool initLastFrameMatrix = false;

            if (transformBuffer == nullptr)
            {
                transformBuffer =
                    rhi->CreateBufferCoherent(sizeof(MeshInstanceTransform), RhiBufferUsage::RhiBufferUsage_SSBO);
                bindlessRef = rhi->RegisterStorageBufferShared(transformBuffer.get());
                transform->SetGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
                initLastFrameMatrix = true;
                transformBufferLast =
                    rhi->CreateBufferCoherent(sizeof(MeshInstanceTransform), RhiBufferUsage::RhiBufferUsage_SSBO);
                bindlessRefLast = rhi->RegisterStorageBufferShared(transformBufferLast.get());
                transform->SetGPUResource(transformBuffer, transformBufferLast, bindlessRef, bindlessRefLast);
            }

            // update uniform buffer, TODO: dirty flag
            auto transformDirty = transform->GetDirtyFlag();
            if (transformDirty.changed || transformDirty.lastChanged)
            {
                MeshInstanceTransform model;
                FillingMeshTransformData(transform, model);

                auto buf = transformBuffer->GetActiveBuffer();
                buf->MapMemory();
                buf->WriteBuffer(&model, sizeof(MeshInstanceTransform), 0);
                buf->FlushBuffer();
                buf->UnmapMemory();

                ret.m_Changed          = true;
                ret.m_TransformRef     = bindlessRef->GetActiveId();
                ret.m_TransformRefLast = bindlessRefLast->GetActiveId();
            }

            if (initLastFrameMatrix || transformDirty.lastChanged)
            {
                MeshInstanceTransform modelLast;
                FillingMeshTransformData(transform, modelLast, true);
                auto bufLast = transformBufferLast->GetActiveBuffer();
                bufLast->MapMemory();
                bufLast->WriteBuffer(&modelLast, sizeof(MeshInstanceTransform), 0);
                bufLast->FlushBuffer();
                bufLast->UnmapMemory();
                transform->OnFrameCollecting();
            }
        }
        else if (updateDevice == TransformUpdateDevice::GPU)
        {
            RhiBuffer* deviceOnlyBuffer = nullptr;
            transform->GetGPUResourceDeviceMode(deviceOnlyBuffer);
            if (deviceOnlyBuffer == nullptr)
            {
                auto bufferUsage = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
                auto buffer = rhi->CreateBufferDevice("Transform", sizeof(MeshInstanceTransform), bufferUsage, true);

                MeshInstanceTransform model;
                FillingMeshTransformData(transform, model);

                auto tq            = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
                auto stagingBuffer = rhi->CreateStagedSingleBuffer(buffer.get());
                tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                    stagingBuffer->CmdCopyToDevice(cmd, &model, sizeof(MeshInstanceTransform), 0);
                });

                transform->SetGPUResourceDeviceMode(buffer);
                deviceOnlyBuffer = buffer.get();    
            }
            ret.m_Changed          = true;
            ret.m_TransformRef     = rhi->GetUAVDescriptor(deviceOnlyBuffer);
            ret.m_TransformRefLast = rhi->GetUAVDescriptor(deviceOnlyBuffer);
        }
        return ret;
    }

} // namespace Ifrit::Runtime