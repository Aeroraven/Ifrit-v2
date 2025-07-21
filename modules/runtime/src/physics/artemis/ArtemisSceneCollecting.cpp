#include "ifrit/runtime/physics/artemis/ArtemisSceneCollecting.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/runtime/scene/FrameComponentUpdate.h"
#include "ifrit/runtime/physics/artemis/rigid/GpuRigidCollider.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit.internal/runtime/physics/artemis/InternalConst.h"

namespace Ifrit::Runtime::Artemis
{
    IFRIT_APIDECL void CollectPhysicsSceneData(Scene* scene, RHI::RhiBackend* rhi)
    {
        // TODO: make this owned by scene (not sharing ownership)
        auto perframeData = scene->GetPerFrameData();
        if (perframeData->m_ExtraData.count(Internal::kArtemisSceneDataKey) == 0)
        {
            perframeData->m_ExtraData[Internal::kArtemisSceneDataKey] = MakeRef<ArtemisSceneData>();
        }
        auto physicsData =
            CheckedPointerCast<ArtemisSceneData>(perframeData->m_ExtraData[Internal::kArtemisSceneDataKey]);

        // find all rigid bodies
        auto rigidObjects = scene->FilterObjects([](GameObject* obj) { return obj->GetComponent<GPURigidCollider>(); });
        auto numRigids    = SizeCast<u32>(rigidObjects.size());

        if ((physicsData->m_NumGpuColliders != numRigids || physicsData->m_GpuColliderDataBuffer == nullptr))
        {
            u32  requiredBufferSize = std::max(1u, SizeCast<u32>(sizeof(ArtemisColliderElement) * numRigids));
            auto bufferUsage = RHI::RhiBufferUsage::RhiBufferUsage_SSBO | RHI::RhiBufferUsage::RhiBufferUsage_CopyDst;
            physicsData->m_GpuColliderDataBuffer =
                rhi->CreateBufferDevice("ArtemisColliderDataBuffer", requiredBufferSize, bufferUsage, true);
        }
        physicsData->m_NumGpuColliders = numRigids;

        // TODO: currently forcing the buffer update
        physicsData->m_ColliderData.resize(numRigids);
        for (u32 i = 0; i < numRigids; ++i)
        {
            auto rigidCollider      = rigidObjects[i]->GetComponent<GPURigidCollider>();
            auto transformComponent = rigidObjects[i]->GetComponent<Transform>();
            IF_LOG_ASSERTION("Artemis.SceneCollecting",
                transformComponent->GetUpdateDevice() == TransformUpdateDevice::GPU,
                "Transform must be in GPU update mode for Artemis rigid collider");
            auto transformRet                             = UpdateTransformGPUData(transformComponent, rhi);
            physicsData->m_ColliderData[i].m_TransformRef = transformRet.m_TransformRef;
            physicsData->m_ColliderData[i].m_Radius       = rigidCollider->GetRadius();

            rigidCollider->OnFrameCollecting();
        }
        auto tq            = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Transfer);
        auto stagingBuffer = rhi->CreateStagedSingleBuffer(physicsData->m_GpuColliderDataBuffer.get());
        tq->RunSyncCommand([&](const RHI::RhiCommandList* cmd) {
            stagingBuffer->CmdCopyToDevice(cmd, physicsData->m_ColliderData.data(),
                SizeCast<u32>(physicsData->m_ColliderData.size() * sizeof(ArtemisColliderElement)), 0);
        });
    }
} // namespace Ifrit::Runtime::Artemis