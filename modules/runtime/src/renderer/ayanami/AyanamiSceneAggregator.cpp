
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "ifrit/runtime/renderer/ayanami/AyanamiSceneAggregator.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiMeshDF.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/core/math/linalg/GeometryFunctions.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::Runtime::Ayanami
{

    struct AyanamiSceneResources
    {
        using GPUTexture     = RHI::RhiTexture;
        using GPUBuffer      = RHI::RhiBuffer;
        using GPUMultiBuffer = RHI::RhiMultiBuffer;
        using GPUBindId      = RHI::RhiDescHandleLegacy;

        struct MDFDescriptor
        {
            u32 m_mdfMetaId;
            u32 m_transformId;
        };
        Vec<MDFDescriptor>                       m_meshMetaIds;

        u32                                      m_m_mdfAllInstancesAllocSize = 0;
        Ref<GPUMultiBuffer>                      m_mdfAllInstances            = nullptr;
        Ref<GPUBindId>                           m_mdfAllInstancesBindId;

        Vector4f                                 m_BoundBall;
        AyanamiSceneAggregator::AggregatedLights m_AggregatedLights;
        Vector3f                                 m_SceneBoundMin;
        Vector3f                                 m_SceneBoundMax;
    };

    IFRIT_APIDECL void AyanamiSceneAggregator::CollectScene(Scene* scene)
    {
        // Here, filter out all objects with mesh df components

        using Ifrit::SizeCast;

        m_sceneResources->m_meshMetaIds.clear();
        f32 minX = std::numeric_limits<f32>::max(), maxX = -std::numeric_limits<f32>::max();
        f32 minY = std::numeric_limits<f32>::max(), maxY = -std::numeric_limits<f32>::max();
        f32 minZ = std::numeric_limits<f32>::max(), maxZ = -std::numeric_limits<f32>::max();
        scene->FilterObjects([&](GameObject* obj) {
            auto       meshDF = obj->GetComponent<AyanamiMeshDF>();
            Vector3f   bboxMin, bboxMax;
            Matrix4x4f modelMat;
            if (meshDF != nullptr)
            {
                // Get transform
                auto transform = obj->GetComponent<Transform>();
                if (transform == nullptr)
                {
                    IF_LOG_CRITICAL(
                        "Ayanami.SceneCollector", "CollectScene() requires transform to be attached to a object");
                }
                // Collect mesh df data
                meshDF->BuildGPUResource(m_rhi);
                auto metaId = meshDF->GetMetaBufferId();
                bboxMax     = meshDF->GetBoxMax();
                bboxMin     = meshDF->GetBoxMin();
                AyanamiSceneResources::MDFDescriptor desc;
                desc.m_mdfMetaId = metaId;
                {
                    using namespace Ifrit::RHI;
                    Ref<RhiMultiBuffer>      transformBuf, transformBufLast;
                    Ref<RhiDescHandleLegacy> transformBindId, transformBindIdLast;
                    transform->GetGPUResource(transformBuf, transformBufLast, transformBindId, transformBindIdLast);
                    desc.m_transformId = transformBindId->GetActiveId();
                    modelMat           = transform->GetModelToWorldMatrix();
                }
                m_sceneResources->m_meshMetaIds.push_back(desc);

                for (int i = 0; i < 8; i++)
                {
                    using namespace Ifrit::Math;
                    f32      x = (i & 1) ? bboxMax.x : bboxMin.x;
                    f32      y = (i & 2) ? bboxMax.y : bboxMin.y;
                    f32      z = (i & 4) ? bboxMax.z : bboxMin.z;
                    Vector4f v = MatMul(modelMat, Vector4f(x, y, z, 1.0f));
                    v          = v / v.w;
                    // printf("v: %f %f %f %f\n", v.x, v.y, v.z, v.w);
                    minX = std::min(minX, v.x);
                    maxX = std::max(maxX, v.x);
                    minY = std::min(minY, v.y);
                    maxY = std::max(maxY, v.y);
                    minZ = std::min(minZ, v.z);
                    maxZ = std::max(maxZ, v.z);
                }
            }

            using namespace Ifrit::Math;
            Vector3f center               = Vector3f((minX + maxX) * 0.5f, (minY + maxY) * 0.5f, (minZ + maxZ) * 0.5f);
            Vector3f size                 = Vector3f(maxX - minX, maxY - minY, maxZ - minZ);
            float    radius               = Length(size) * 0.5f;
            m_sceneResources->m_BoundBall = Vector4f(center.x, center.y, center.z, radius);
            m_sceneResources->m_SceneBoundMin = Vector3f(minX, minY, minZ);
            m_sceneResources->m_SceneBoundMax = Vector3f(maxX, maxY, maxZ);

            // TODO: non-directional light
            AggregatedLights  lights;
            PerFrameData*     perFrameData = scene->GetPerFrameData();
            for (u32 i = 0; auto& v : perFrameData->m_shadowData2.m_LightFronts)
            {
                lights.m_LightFronts.push_back(v);
                if ((++i) >= perFrameData->m_shadowData2.m_enabledShadowMaps)
                    break;
            }
            m_sceneResources->m_AggregatedLights = lights;
            return false;
        });

        // All instances gathered, now build the all instance buffer
        if (m_sceneResources->m_mdfAllInstances != nullptr
            && m_sceneResources->m_m_mdfAllInstancesAllocSize != m_sceneResources->m_meshMetaIds.size())
        {
            IF_LOG_CRITICAL("Ayanami.SceneCollector", "CollectScene() does not support dynamic scene now");
            std::abort();
        }

        if (m_sceneResources->m_mdfAllInstances == nullptr)
        {
            m_sceneResources->m_m_mdfAllInstancesAllocSize = SizeCast<u32>(m_sceneResources->m_meshMetaIds.size());
            m_sceneResources->m_mdfAllInstances            = m_rhi->CreateBufferCoherent(
                sizeof(AyanamiSceneResources::MDFDescriptor) * m_sceneResources->m_m_mdfAllInstancesAllocSize,
                RHI::RhiBufferUsage::RhiBufferUsage_CopyDst | RHI::RhiBufferUsage::RhiBufferUsage_SSBO);
        }

        // update the buffer
        auto activeBuf = m_sceneResources->m_mdfAllInstances->GetActiveBuffer();
        activeBuf->MapMemory();
        activeBuf->WriteBuffer(m_sceneResources->m_meshMetaIds.data(),
            SizeCast<u32>(m_sceneResources->m_meshMetaIds.size() * sizeof(AyanamiSceneResources::MDFDescriptor)), 0);
        activeBuf->FlushBuffer();
        activeBuf->UnmapMemory();

        // set id
        m_sceneResources->m_mdfAllInstancesBindId =
            m_rhi->RegisterStorageBufferShared(m_sceneResources->m_mdfAllInstances.get());
    }

    IFRIT_APIDECL u32 AyanamiSceneAggregator::GetGatheredBufferId()
    {
        return m_sceneResources->m_mdfAllInstancesBindId->GetActiveId();
    }

    IFRIT_APIDECL u32 AyanamiSceneAggregator::GetNumGatheredInstances() const
    {
        return SizeCast<u32>(m_sceneResources->m_meshMetaIds.size());
    }

    IFRIT_APIDECL void AyanamiSceneAggregator::Init() { m_sceneResources = new AyanamiSceneResources(); }

    IFRIT_APIDECL void AyanamiSceneAggregator::Destroy()
    {
        if (m_sceneResources != nullptr)
        {
            delete m_sceneResources;
            m_sceneResources = nullptr;
        }
    }

    IFRIT_APIDECL Vector4f AyanamiSceneAggregator::GetSceneBoundSphere() const { return m_sceneResources->m_BoundBall; }

    IFRIT_APIDECL AyanamiSceneAggregator::AggregatedLights AyanamiSceneAggregator::GetAggregatedLights() const
    {
        return m_sceneResources->m_AggregatedLights;
    }

    IFRIT_APIDECL Matrix4x4f AyanamiSceneAggregator::GetLightBoundVP(Vector3f lightDirWS) const
    {
        return Math::GetViewOrthoProjectionForAABB(
            m_sceneResources->m_SceneBoundMin, m_sceneResources->m_SceneBoundMax, lightDirWS);
    }

    IFRIT_APIDECL Vector3f AyanamiSceneAggregator::GetSceneBoundMin() const
    {
        return m_sceneResources->m_SceneBoundMin;
    }
    IFRIT_APIDECL Vector3f AyanamiSceneAggregator::GetSceneBoundMax() const
    {
        return m_sceneResources->m_SceneBoundMax;
    }

} // namespace Ifrit::Runtime::Ayanami