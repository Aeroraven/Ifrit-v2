
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

#include "ifrit/core/renderer/ayanami/AyanamiTrivialSurfaceCache.h"
#include "ifrit/core/renderer/ayanami/AyanamiMeshMarker.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/base/Mesh.h"
#include "ifrit/core/material/SyaroDefaultGBufEmitter.h"
#include "ifrit/core/renderer/util/RenderingUtils.h"

using namespace Ifrit::Graphics::Rhi;

namespace Ifrit::Core::Ayanami
{
    static constexpr Array<Vector3f, 6> kCardDirections = {
        Vector3f(1.0f, 0.0f, 0.0f), Vector3f(-1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f),
        Vector3f(0.0f, -1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f), Vector3f(0.0f, 0.0f, -1.0f)
    };

    static constexpr Array<Vector3f, 6> kCardLookAtUps = {
        Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(1.0f, 0.0f, 0.0f),
        Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f)
    };

    // 3 Directions:
    // View From +x/-x: (Z,Y,X)
    // View From +y/-y: (Z,X,Y)
    // View From +z/-z: (X,Y,Z)
    static constexpr Array<Vector3i, 3> kCardViewAxis = {
        Vector3i(2, 1, 0), Vector3i(2, 0, 1), Vector3i(0, 1, 2)
    };

    struct ManagedMeshCard
    {
        SceneObject* m_Object = nullptr;
        Vector3f     m_CardDirection;
        Vector3f     m_ObjectScale;
        Vector2u     m_CardLocation;
        Vector2u     m_CardExtent;
        RhiBufferRef m_CardVertexBuffer;
        RhiBufferRef m_CardIndexBuffer;
        RhiBufferRef m_CardUVBuffer;
        u32          m_ObjectBufferId;
        u32          m_IndexCounts;

        Matrix4x4f   m_ObserverView;
        Matrix4x4f   m_ObserverProj;
        Matrix4x4f   m_ObserverVP;

        u32          m_TempAlbedoId;
        u32          m_TempNormalId;
    };

    struct ManagedMeshCardGPUData
    {
        Matrix4x4f m_ObserverVP;
    };

    struct AyanamiTrivialSurfaceCacheManagerResource
    {
        bool                        m_Inited             = false;
        bool                        m_RequireGpuDataSync = false;

        RhiTextureRef               m_SceneCacheAlbdeoAtlas;
        RhiTextureRef               m_SceneCacheNormalAtlas;
        RhiTextureRef               m_SceneCacheEmissionAtlas;
        RhiTextureRef               m_SceneCacheSpecularAtlas;
        Atomic<u32>                 m_MeshCardIndex = 0;

        Vec<ManagedMeshCard>        m_MeshCards;
        Vec<u32>                    m_MeshCardTasks;
        Vec<ManagedMeshCardGPUData> m_MeshCardGPUData;

        // Here, mipmaps will be considered later. Now, we only use 1 mipmap level.
        u32                         m_AtlasElementSize    = 64;
        u32                         m_CurrentAtlasElement = 0;

        // TODO: this stores the matrixs for the observer view;
        // I hope it to be uniform for compatibility. However, the data is too large.
        // So, we need to use a storage buffer to store the data.
        RhiBufferRef                m_ObserveDeviceData;

        Ref<RhiVertexBufferView>    m_SurfaceCachePassBinding;
        RhiGraphicsPass*            m_SurfaceCachePass = nullptr;

        Ref<RhiRenderTargets>       m_SurfaceCachePassRTs;
        Ref<RhiColorAttachment>     m_SurfaceCachePassAlbedoRT;

        // Debug Controls
        bool                        m_ForceSurfaceCacheRegeneration = false;
    };

    AyanamiTrivialSurfaceCacheManager::AyanamiTrivialSurfaceCacheManager(const AyanamiRenderConfig& config, IApplication* app)
        : m_App(app), m_Resolution(config.m_surfaceCacheResolution)
    {
        m_Resources                                  = new AyanamiTrivialSurfaceCacheManagerResource();
        m_Resources->m_ForceSurfaceCacheRegeneration = config.m_DebugForceSurfaceCacheRegen;

        PrepareImmutableResource();
    }
    AyanamiTrivialSurfaceCacheManager::~AyanamiTrivialSurfaceCacheManager()
    {
        delete m_Resources;
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateSceneCache(Scene* scene)
    {
        using namespace Ifrit::Math;

        m_Resources->m_MeshCardTasks.clear();
        auto objects = scene->FilterObjectsUnsafe([](SceneObject* obj) {
            return obj->GetComponent<AyanamiMeshMarker>() != nullptr;
        });

        for (auto obj : objects)
        {
            auto marker       = obj->GetComponent<AyanamiMeshMarker>();
            auto meshFilter   = obj->GetComponent<MeshFilter>();
            auto meshRenderer = obj->GetComponent<MeshRenderer>();

            if (meshFilter == nullptr)
            {
                iError("MeshFilter is nullptr");
                std::abort();
            }

            if (meshRenderer == nullptr)
            {
                iError("MeshRenderer is nullptr");
                std::abort();
            }

            auto material = meshRenderer->GetMaterial();
            if (material == nullptr)
            {
                iError("Material is nullptr");
                std::abort();
            }

            // TODO: Batcher should be used to batch the meshes with same material
            // for simplicity, we just use the default material
            using Ifrit::Common::Utility::CheckedPointerCast;
            auto castedMaterial = CheckedPointerCast<SyaroDefaultGBufEmitter>(material);
            auto albedoId       = castedMaterial->GetAlbedoId();
            auto normalId       = castedMaterial->GetNormalMapId();

            auto meshCardId = marker->GetTrivialMeshCardIndex();
            if (meshCardId == ~0u)
            {
                // No mesh card assigned, allocate one
                auto allocId = m_Resources->m_MeshCardIndex.fetch_add(6);
                marker->SetTrivialMeshCardIndex(allocId);

                auto meshWrapper  = meshFilter->GetMesh();
                auto meshData     = meshWrapper->LoadMesh();
                auto meshResource = Mesh::GPUResource();
                meshWrapper->GetGPUResource(meshResource);

                auto vertexBuffer = meshResource.vertexBuffer;
                auto indexBuffer  = meshResource.indexBuffer;
                auto uvBuffer     = meshResource.uvBuffer;
                auto indexCounts  = meshData->m_indices.size();

                if (vertexBuffer == nullptr || indexBuffer == nullptr)
                {
                    iError("Vertex buffer or index buffer is nullptr");
                    std::abort();
                }

                auto          objectBufferId = meshResource.objectBuffer->GetDescId();
                auto          meshBBoxMax    = meshData->m_BoundingBoxMax;
                auto          meshBBoxMin    = meshData->m_BoundingBoxMin;
                auto          meshBBoxCenter = (meshBBoxMax + meshBBoxMin) * 0.5f;
                auto          meshBBoxSize   = meshBBoxMax - meshBBoxMin;
                auto          meshBBoxExtent = meshBBoxSize * 0.5f;

                Array<f32, 3> meshBBoxExtentArr = { meshBBoxExtent.x, meshBBoxExtent.y, meshBBoxExtent.z };

                printf("Mesh Extent: %f %f %f\n", meshBBoxExtent.x, meshBBoxExtent.y, meshBBoxExtent.z);

                ManagedMeshCard card;
                for (u32 i = 0; i < 6; i++)
                {
                    auto slotId             = allocId + i;
                    card.m_Object           = obj;
                    card.m_CardDirection    = kCardDirections[i];
                    card.m_ObjectScale      = obj->GetComponent<Transform>()->GetScale();
                    card.m_CardVertexBuffer = vertexBuffer;
                    card.m_CardIndexBuffer  = indexBuffer;
                    card.m_IndexCounts      = indexCounts;
                    card.m_CardUVBuffer     = uvBuffer;
                    card.m_ObjectBufferId   = objectBufferId;

                    auto elementsPerRow   = m_Resolution / m_Resources->m_AtlasElementSize;
                    auto index_X          = slotId % elementsPerRow;
                    auto index_Y          = slotId / elementsPerRow;
                    card.m_CardLocation.x = index_X * m_Resources->m_AtlasElementSize;
                    card.m_CardLocation.y = index_Y * m_Resources->m_AtlasElementSize;

                    // Generate Matrices for Rasterization
                    Vector3f cardExtent = Vector3f(0.0f, 0.0f, 0.0f);
                    cardExtent.x        = meshBBoxExtentArr[kCardViewAxis[i / 2].x];
                    cardExtent.y        = meshBBoxExtentArr[kCardViewAxis[i / 2].y];
                    cardExtent.z        = meshBBoxExtentArr[kCardViewAxis[i / 2].z];

                    card.m_CardExtent.x = m_Resources->m_AtlasElementSize;
                    card.m_CardExtent.y = m_Resources->m_AtlasElementSize;

                    // LookAt & Ortho
                    f32        viewNearPlane = 0.1f;
                    Vector3f   viewLocation  = meshBBoxCenter - card.m_CardDirection * cardExtent - viewNearPlane;
                    Vector3f   viewUp        = kCardLookAtUps[i];
                    Vector3f   viewTarget    = meshBBoxCenter;
                    Matrix4x4f viewMatrix    = LookAt(viewLocation, viewTarget, viewUp);

                    f32        viewAspect = cardExtent.x / cardExtent.y;
                    Matrix4x4f viewOrtho  = OrthographicNegateY(cardExtent.y, viewAspect, viewNearPlane, cardExtent.z * 2.0f + viewNearPlane);
                    Matrix4x4f viewProj   = MatMul(viewOrtho, viewMatrix);

                    Matrix4x4f viewVP   = MatMul(viewProj, viewMatrix);
                    card.m_ObserverView = viewMatrix;
                    card.m_ObserverProj = viewOrtho;
                    card.m_ObserverVP   = viewVP;

                    card.m_TempAlbedoId = albedoId;
                    card.m_TempNormalId = normalId;

                    m_Resources->m_MeshCardGPUData[slotId].m_ObserverVP = viewVP;
                    m_Resources->m_MeshCards[slotId]                    = card;
                    m_Resources->m_RequireGpuDataSync                   = true;

                    m_Resources->m_MeshCardTasks.push_back(slotId);
                }
            }
        }

        if (m_Resources->m_RequireGpuDataSync)
        {
            using namespace Ifrit::Common::Utility;

            auto tq           = m_App->GetRhi()->GetQueue(RhiQueueCapability::RHI_QUEUE_TRANSFER_BIT);
            auto stagedBuffer = m_App->GetRhi()->CreateStagedSingleBuffer(m_Resources->m_ObserveDeviceData.get());
            tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                stagedBuffer->CmdCopyToDevice(cmd, m_Resources->m_MeshCardGPUData.data(),
                    SizeCast<u32>(m_Resources->m_MeshCardGPUData.size() * sizeof(ManagedMeshCardGPUData)), 0);
            });
            m_Resources->m_RequireGpuDataSync = false;
        }

        // If forced regen
        if (m_Resources->m_ForceSurfaceCacheRegeneration)
        {
            m_Resources->m_MeshCardTasks.clear();
            auto totalIndexCount = m_Resources->m_MeshCardIndex.load();
            for (u32 i = 0; i < totalIndexCount; i++)
            {
                m_Resources->m_MeshCardTasks.push_back(i);
            }
        }
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateSurfaceCacheAtlas(const Graphics::Rhi::RhiCommandList* cmdList)
    {
        // Albedo , for simplicity
        auto rhi = m_App->GetRhi();
        m_Resources->m_SurfaceCachePass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            auto cmd = ctx->m_cmd;

            for (auto id : m_Resources->m_MeshCardTasks)
            {
                RhiViewport      viewport;
                ManagedMeshCard& card = m_Resources->m_MeshCards[id];
                viewport.x            = card.m_CardLocation.x;
                viewport.y            = card.m_CardLocation.y;
                viewport.width        = card.m_CardExtent.x;
                viewport.height       = card.m_CardExtent.y;
                viewport.minDepth     = 0.0f;
                viewport.maxDepth     = 1.0f;
                cmd->SetViewports({ viewport });

                RhiScissor scissor;
                scissor.x      = card.m_CardLocation.x;
                scissor.y      = card.m_CardLocation.y;
                scissor.width  = card.m_CardExtent.x;
                scissor.height = card.m_CardExtent.y;
                cmd->SetScissors({ scissor });

                // cmd->AttachVertexBufferView(*m_Resources->m_SurfaceCachePassBinding.get());
                // cmd->AttachVertexBuffers(0, { card.m_CardVertexBuffer.get() });
                cmd->AttachIndexBuffer(card.m_CardIndexBuffer.get());

                struct PushConst
                {
                    u32 albedoId;
                    u32 normalId;
                    u32 objectId;
                    u32 cardId;
                    u32 vertexId;
                    u32 uvId;
                    u32 allCardDataId;
                } pc;

                pc.albedoId      = card.m_TempAlbedoId;
                pc.normalId      = card.m_TempNormalId;
                pc.objectId      = card.m_ObjectBufferId;
                pc.cardId        = id;
                pc.vertexId      = card.m_CardVertexBuffer->GetDescId();
                pc.uvId          = card.m_CardUVBuffer->GetDescId();
                pc.allCardDataId = m_Resources->m_ObserveDeviceData->GetDescId();

                cmd->SetPushConst(m_Resources->m_SurfaceCachePass, 0, sizeof(PushConst), &pc);
                cmd->DrawIndexed(card.m_IndexCounts, 1, 0, 0, 0);
            }
        });

        cmdList->BeginScope("Ayanami: SurfaceCacheGenPass");
        m_Resources->m_SurfaceCachePass->Run(cmdList, m_Resources->m_SurfaceCachePassRTs.get(), 0);
        cmdList->EndScope();
    }

    IFRIT_APIDECL void
    AyanamiTrivialSurfaceCacheManager::PrepareImmutableResource()
    {
        if (m_Resources->m_Inited)
            return;
        auto rhi                             = m_App->GetRhi();
        m_Resources->m_SceneCacheAlbdeoAtlas = rhi->CreateTexture2D(
            "AyanamiTrivialSurfaceCache_AlbedoAtlas", m_Resolution, m_Resolution,
            RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true);
        m_Resources->m_SceneCacheNormalAtlas = rhi->CreateTexture2D(
            "AyanamiTrivialSurfaceCache_NormalAtlas", m_Resolution, m_Resolution,
            RhiImageFormat::RhiImgFmt_R8G8B8A8_SNORM,
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true);
        m_Resources->m_SceneCacheEmissionAtlas = rhi->CreateTexture2D(
            "AyanamiTrivialSurfaceCache_EmissionAtlas", m_Resolution, m_Resolution,
            RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true);
        m_Resources->m_SceneCacheSpecularAtlas = rhi->CreateTexture2D(
            "AyanamiTrivialSurfaceCache_SpecularAtlas", m_Resolution, m_Resolution,
            RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, true);

        auto maxAtlasSlots              = m_Resolution * m_Resolution / m_Resources->m_AtlasElementSize / m_Resources->m_AtlasElementSize;
        auto requiredObserverBufferSize = maxAtlasSlots * sizeof(ManagedMeshCardGPUData);

        m_Resources->m_ObserveDeviceData = rhi->CreateBuffer(
            "AyanamiTrivialSurfaceCache_ObserverData", requiredObserverBufferSize,
            RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst, true, true);
        m_Resources->m_MeshCardGPUData.resize(maxAtlasSlots);
        m_Resources->m_MeshCards.resize(maxAtlasSlots);

        // Then passes configs
        m_Resources->m_SurfaceCachePassBinding = rhi->CreateVertexBufferView();
        m_Resources->m_SurfaceCachePassBinding->AddBinding({ 0 }, { RhiImageFormat::RhiImgFmt_R32G32B32_SFLOAT },
            { 0 }, 3 * sizeof(float));

        using Ifrit::Core::RenderingUtil::CreateGraphicsPass;

        RhiRenderTargetsFormat rtFmt;
        rtFmt.m_colorFormats.push_back(RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM);

        m_Resources->m_SurfaceCachePass = CreateGraphicsPass(rhi, "Ayanami/Ayanami.SurfaceCacheGen.vert.glsl",
            "Ayanami/Ayanami.SurfaceCacheGen.frag.glsl", 0, 7, rtFmt);

        // RTs
        m_Resources->m_SurfaceCachePassRTs      = rhi->CreateRenderTargets();
        m_Resources->m_SurfaceCachePassAlbedoRT = rhi->CreateRenderTarget(
            m_Resources->m_SceneCacheAlbdeoAtlas.get(), { 0, 0, 1, 1 },
            RhiRenderTargetLoadOp::Load, 0, 0);
        m_Resources->m_SurfaceCachePassRTs->SetColorAttachments({ m_Resources->m_SurfaceCachePassAlbedoRT.get() });

        m_Resources->m_SurfaceCachePass->SetRenderTargetFormat(m_Resources->m_SurfaceCachePassRTs->GetFormat());
        m_Resources->m_SurfaceCachePassRTs->SetRenderArea({ 0, 0, m_Resolution, m_Resolution });
        m_Resources->m_Inited = true;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiTextureRef AyanamiTrivialSurfaceCacheManager::GetAlbedoAtlas()
    {
        return m_Resources->m_SceneCacheAlbdeoAtlas;
    }
} // namespace Ifrit::Core::Ayanami