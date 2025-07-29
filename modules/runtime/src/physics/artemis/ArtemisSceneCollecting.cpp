#include "ifrit/runtime/physics/artemis/ArtemisSceneCollecting.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/runtime/scene/FrameComponentUpdate.h"
#include "ifrit/runtime/physics/artemis/rigid/GpuRigidCollider.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit.internal/runtime/physics/artemis/InternalConst.h"
#include "ifrit.shader.neo/Shared/Artemis/Rigid.Shared.h"
#include <algorithm>

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
        auto rigidObjects          = scene->FilterObjects([](GameObject* obj) {
            auto fv = obj->GetComponent<GPURigidCollider>();
            return fv ? fv->IsEnabled() : false;
        });
        auto numRigids             = SizeCast<u32>(rigidObjects.size());
        bool shouldInitRuntimeData = false;

        if (physicsData->m_GpuColliderDataBufferRuntime == nullptr)
        {
            shouldInitRuntimeData = true;
            u32 requiredBufferSize =
                std::max(1u, SizeCast<u32>(sizeof(ArtemisColliderElementRuntimeData) * Internal::kArtemisMaxColliders));
            auto bufferUsage = RHI::RhiBufferUsage::RhiBufferUsage_SSBO | RHI::RhiBufferUsage::RhiBufferUsage_CopyDst;
            physicsData->m_GpuColliderDataBufferRuntime =
                rhi->CreateBufferDevice("ArtemisColliderDataBufferRuntime", requiredBufferSize, bufferUsage, true);
        }

        if ((physicsData->m_GpuColliderDataBuffer == nullptr))
        {
            u32 requiredBufferSize = std::max(
                1u, SizeCast<u32>(sizeof(Shader::Artemis::FRigidColliderEntry) * Internal::kArtemisMaxColliders));
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
            auto transformRet                                   = UpdateTransformGPUData(transformComponent, rhi);
            physicsData->m_ColliderData[i].m_Transform          = transformRet.m_TransformRef;
            physicsData->m_ColliderData[i].m_ColliderRadius     = rigidCollider->GetRadius();
            physicsData->m_ColliderData[i].m_RigidMass          = rigidCollider->GetRigidMass();
            physicsData->m_ColliderData[i].m_ColliderCuboidSize = Vector4f(rigidCollider->GetCuboidSize(), 0.0f);
            physicsData->m_ColliderData[i].m_ColliderType       = rigidCollider->GetColliderType();
            physicsData->m_ColliderData[i].m_Inertia2D          = rigidCollider->GetMomentOfInertia2D();

            if (rigidCollider->GetInternalRigidId() == ~0u)
            {
                rigidCollider->SetInternalRigidId((physicsData->m_AllocatedRuntimeIds++) + 5);
            }
            physicsData->m_ColliderData[i].m_RuntimeId = rigidCollider->GetInternalRigidId();

            rigidCollider->OnFrameCollecting();
        }
        std::sort(physicsData->m_ColliderData.begin(), physicsData->m_ColliderData.end(),
            [](const Shader::Artemis::FRigidColliderEntry& a, const Shader::Artemis::FRigidColliderEntry& b) {
                return a.m_RuntimeId < b.m_RuntimeId;
            });

        auto tq            = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Transfer);
        auto stagingBuffer = rhi->CreateStagedSingleBuffer(physicsData->m_GpuColliderDataBuffer.get());
        // rhi->WaitDeviceIdle();
        tq->RunSyncCommand([&](const RHI::RhiCommandList* cmd) {
            if (physicsData->m_ColliderData.size())
            {
                stagingBuffer->CmdCopyToDevice(cmd, physicsData->m_ColliderData.data(),
                    SizeCast<u32>(physicsData->m_ColliderData.size() * sizeof(Shader::Artemis::FRigidColliderEntry)),
                    0);
                if (shouldInitRuntimeData)
                {
                    cmd->BufferClear(physicsData->m_GpuColliderDataBufferRuntime.get(), 0);
                }
            }
        });
        // rhi->WaitDeviceIdle();
    }
} // namespace Ifrit::Runtime::Artemis
