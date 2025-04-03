
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

#include "ifrit/runtime/renderer/ayanami/AyanamiTrivialSurfaceCache.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiMeshMarker.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/runtime/material/SyaroDefaultGBufEmitter.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"

#include "ifrit.shader/Ayanami/Ayanami.SharedConst.h"

#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.h"

using namespace Ifrit::Graphics::Rhi;
using Ifrit::Math::DivRoundUp;

namespace Ifrit::Runtime::Ayanami
{
    static constexpr Array<Vector3f, 6> kCardDirections = { Vector3f(1.0f, 0.0f, 0.0f), Vector3f(-1.0f, 0.0f, 0.0f),
        Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, -1.0f, 0.0f), Vector3f(0.0f, 0.0f, 1.0f),
        Vector3f(0.0f, 0.0f, -1.0f) };

    static constexpr Array<Vector3f, 6> kCardLookAtUps = { Vector3f(0.0f, 1.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f),
        Vector3f(1.0f, 0.0f, 0.0f), Vector3f(1.0f, 0.0f, 0.0f), Vector3f(0.0f, 1.0f, 0.0f),
        Vector3f(0.0f, 1.0f, 0.0f) };

    // 3 Directions:
    // View From +x/-x: (Z,Y,X)
    // View From +y/-y: (Z,X,Y)
    // View From +z/-z: (X,Y,Z)
    static constexpr Array<Vector3i, 3> kCardViewAxis = { Vector3i(2, 1, 0), Vector3i(2, 0, 1), Vector3i(0, 1, 2) };

    struct ManagedMeshCard
    {
        GameObject*  m_Object = nullptr;
        Vector3f     m_CardDirection;
        Vector3f     m_ObjectScale;
        Vector2u     m_CardLocation;
        Vector2u     m_CardExtent;
        RhiBufferRef m_CardVertexBuffer;
        RhiBufferRef m_CardIndexBuffer;
        RhiBufferRef m_CardUVBuffer;
        RhiBufferRef m_CardTangentBuffer;
        RhiBufferRef m_CardNormalBuffer;
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
        Matrix4x4f m_ObserverVPInverse;
    };

    struct ManagedMeshCardCoherentGPUData
    {
        u32 m_TransformId;
    };

    struct AyanamiTrivialSurfaceCacheManagerResource
    {
        using GPUBindId = Graphics::Rhi::RhiDescHandleLegacy;

        bool                                m_Inited             = false;
        bool                                m_RequireGpuDataSync = false;

        RhiSamplerRef                       m_CommonSampler;

        RhiTextureRef                       m_SceneCacheAlbdeoAtlas;
        RhiTextureRef                       m_SceneCacheNormalAtlas;
        RhiTextureRef                       m_SceneCacheEmissionAtlas;
        RhiTextureRef                       m_SceneCacheSpecularAtlas;
        RhiTextureRef                       m_SceneCacheTemporaryDepth;
        RhiTextureRef                       m_SceneCacheRadianceAtlas;

        // This marks whether a texel (thread group) on surface cache should use
        // offline shadow map or not.
        RhiBufferRef                        m_ShadowMaskOfflineBuffer;

        Ref<GPUBindId>                      m_SceneCacheDepthSRV;

        Atomic<u32>                         m_MeshCardIndex = 0;

        Vec<ManagedMeshCard>                m_MeshCards;
        Vec<u32>                            m_MeshCardTasks;
        Vec<ManagedMeshCardGPUData>         m_MeshCardGPUData;
        Vec<ManagedMeshCardCoherentGPUData> m_MeshCardCoherentGPUData;

        // Here, mipmaps will be considered later. Now, we only use 1 mipmap level.
        u32                                 m_AtlasElementSize    = 64;
        u32                                 m_CurrentAtlasElement = 0;
        u32                                 m_MaxPerTileLights    = 8;

        // TODO: this stores the matrixs for the observer view;
        // I hope it to be uniform for compatibility. However, the data is too large.
        // So, we need to use a storage buffer to store the data.
        RhiBufferRef                        m_ObserveDeviceData;
        Ref<RhiMultiBuffer>                 m_ObserveDeviceDataCoherent;
        Ref<GPUBindId>                      m_ObserveDeviceDataCoherentBindId;

        Ref<RhiVertexBufferView>            m_SurfaceCachePassBinding;
        RhiGraphicsPass*                    m_SurfaceCachePass = nullptr;

        Ref<RhiRenderTargets>               m_SurfaceCachePassRTs;
        Ref<RhiColorAttachment>             m_SurfaceCachePassAlbedoRT;
        Ref<RhiColorAttachment>             m_SurfaceCachePassNormalRT;
        Ref<RhiDepthStencilAttachment>      m_SurfaceCachePassDepthRT;

        // Radiance pass (direct lighting)
        RhiComputePass*                     m_RadianceCachePass = nullptr;

        // Debug Controls
        bool                                m_ForceSurfaceCacheRegeneration = false;
    };

    AyanamiTrivialSurfaceCacheManager::AyanamiTrivialSurfaceCacheManager(
        const AyanamiRenderConfig& config, IApplication* app)
        : m_App(app), m_Resolution(config.m_SurfaceCacheResolution)
    {
        m_Resources                                  = new AyanamiTrivialSurfaceCacheManagerResource();
        m_Resources->m_ForceSurfaceCacheRegeneration = config.m_DebugForceSurfaceCacheRegen;

        m_Resources->m_MaxPerTileLights = config.m_RadiancePassMaxPerTileLights;
        PrepareImmutableResource();
    }
    AyanamiTrivialSurfaceCacheManager::~AyanamiTrivialSurfaceCacheManager() { delete m_Resources; }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateSceneCache(Scene* scene)
    {
        using namespace Ifrit::Math;

        m_Resources->m_MeshCardTasks.clear();
        auto objects = scene->FilterObjectsUnsafe(
            [](GameObject* obj) { return obj->GetComponent<AyanamiMeshMarker>() != nullptr; });

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
            using Ifrit::CheckedPointerCast;
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

                auto vertexBuffer  = meshResource.vertexBuffer;
                auto indexBuffer   = meshResource.indexBuffer;
                auto uvBuffer      = meshResource.uvBuffer;
                auto tangentBuffer = meshResource.tangentBuffer;
                auto normalBuffer  = meshResource.normalBuffer;
                auto indexCounts   = meshData->m_indices.size();

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

                auto&         meshVertices = meshData->m_verticesAligned;

                Array<f32, 3> meshBBoxExtentArr = { meshBBoxExtent.x, meshBBoxExtent.y, meshBBoxExtent.z };

                printf("Mesh Extent: %f %f %f\n", meshBBoxExtent.x, meshBBoxExtent.y, meshBBoxExtent.z);
                printf("Mesh Center: %f %f %f\n", meshBBoxCenter.x, meshBBoxCenter.y, meshBBoxCenter.z);
                ManagedMeshCard card;
                for (u32 i = 0; i < 6; i++)
                {
                    auto slotId              = allocId + i;
                    card.m_Object            = obj;
                    card.m_CardDirection     = kCardDirections[i];
                    card.m_ObjectScale       = obj->GetComponent<Transform>()->GetScale();
                    card.m_CardVertexBuffer  = vertexBuffer;
                    card.m_CardIndexBuffer   = indexBuffer;
                    card.m_CardTangentBuffer = tangentBuffer;
                    card.m_CardNormalBuffer  = normalBuffer;

                    card.m_IndexCounts    = indexCounts;
                    card.m_CardUVBuffer   = uvBuffer;
                    card.m_ObjectBufferId = objectBufferId;

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
                    f32      viewNearPlane = 0.1f;
                    Vector3f viewLocation =
                        meshBBoxCenter - card.m_CardDirection * cardExtent - card.m_CardDirection * viewNearPlane;

                    Vector3f   viewUp     = kCardLookAtUps[i];
                    Vector3f   viewTarget = meshBBoxCenter;
                    Matrix4x4f viewMatrix = LookAt(viewLocation, viewTarget, viewUp);

                    f32        viewAspect = cardExtent.x / cardExtent.y;
                    Matrix4x4f viewOrtho  = OrthographicNegateY(
                        cardExtent.y * 2.0, viewAspect, viewNearPlane, cardExtent.z * 2.0f + viewNearPlane);

                    Matrix4x4f viewProj = viewOrtho;
                    Matrix4x4f viewVP   = MatMul(viewProj, viewMatrix);
                    card.m_ObserverView = viewMatrix;
                    card.m_ObserverProj = viewOrtho;
                    card.m_ObserverVP   = viewVP;

                    card.m_TempAlbedoId = albedoId;
                    card.m_TempNormalId = normalId;

                    m_Resources->m_MeshCardGPUData[slotId].m_ObserverVP        = Transpose(viewVP);
                    m_Resources->m_MeshCardGPUData[slotId].m_ObserverVPInverse = Transpose(Inverse4(viewVP));

                    m_Resources->m_MeshCards[slotId]  = card;
                    m_Resources->m_RequireGpuDataSync = true;

                    m_Resources->m_MeshCardTasks.push_back(slotId);
                    // std::abort();
                }
            }
        }

        if (m_Resources->m_RequireGpuDataSync)
        {
            using namespace Ifrit;

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

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateSurfaceCacheAtlas(
        const Graphics::Rhi::RhiCommandList* cmdList)
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
                    u32 normalTexId;
                    u32 objectId;
                    u32 cardId;
                    u32 vertexId;
                    u32 uvId;
                    u32 allCardDataId;
                    u32 tangentId;
                    u32 normalId;
                } pc;

                pc.albedoId      = card.m_TempAlbedoId;
                pc.normalTexId   = card.m_TempNormalId;
                pc.objectId      = card.m_ObjectBufferId;
                pc.cardId        = id;
                pc.vertexId      = card.m_CardVertexBuffer->GetDescId();
                pc.uvId          = card.m_CardUVBuffer->GetDescId();
                pc.allCardDataId = m_Resources->m_ObserveDeviceData->GetDescId();
                pc.tangentId     = card.m_CardTangentBuffer->GetDescId();
                pc.normalId      = card.m_CardNormalBuffer->GetDescId();

                cmd->SetPushConst(m_Resources->m_SurfaceCachePass, 0, sizeof(PushConst), &pc);
                cmd->DrawIndexed(card.m_IndexCounts, 1, 0, 0, 0);
            }
        });

        cmdList->BeginScope("Ayanami: SurfaceCacheGenPass");
        m_Resources->m_SurfaceCachePass->Run(cmdList, m_Resources->m_SurfaceCachePassRTs.get(), 0);
        cmdList->EndScope();
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::PrepareImmutableResource()
    {
        if (m_Resources->m_Inited)
            return;
        auto rhi                     = m_App->GetRhi();
        m_Resources->m_CommonSampler = rhi->CreateTrivialBilinearSampler(true);

        m_Resources->m_SceneCacheAlbdeoAtlas   = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_AlbedoAtlas",
              m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
              RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT
                  | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              true);
        m_Resources->m_SceneCacheNormalAtlas   = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_NormalAtlas",
              m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8G8B8A8_SNORM,
              RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT
                  | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
              true);
        m_Resources->m_SceneCacheEmissionAtlas = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_EmissionAtlas",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT
                | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            true);
        m_Resources->m_SceneCacheSpecularAtlas = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_SpecularAtlas",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT
                | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            true);
        m_Resources->m_SceneCacheRadianceAtlas = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_RadianceAtlas",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8_UNORM,
            RhiImageUsage::RHI_IMAGE_USAGE_STORAGE_BIT | RhiImageUsage::RHI_IMAGE_USAGE_SAMPLED_BIT
                | RhiImageUsage::RHI_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            true);

        // A depth buffer is required for the surface cache pass
        m_Resources->m_SceneCacheTemporaryDepth =
            rhi->CreateDepthTexture("AyanamiTrivialSurfaceCache_DepthAtlas", m_Resolution, m_Resolution, false);

        // Shader read views
        m_Resources->m_SceneCacheDepthSRV = rhi->RegisterCombinedImageSampler(
            m_Resources->m_SceneCacheTemporaryDepth.get(), m_Resources->m_CommonSampler.get());

        auto maxAtlasSlots =
            m_Resolution * m_Resolution / m_Resources->m_AtlasElementSize / m_Resources->m_AtlasElementSize;
        auto requiredObserverBufferSize = maxAtlasSlots * sizeof(ManagedMeshCardGPUData);
        auto requiredCoherentBufferSize = maxAtlasSlots * sizeof(ManagedMeshCardCoherentGPUData);

        m_Resources->m_ObserveDeviceData =
            rhi->CreateBuffer("AyanamiTrivialSurfaceCache_ObserverData", requiredObserverBufferSize,
                RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst, true, true);

        m_Resources->m_ObserveDeviceDataCoherent = rhi->CreateBufferCoherent(
            requiredCoherentBufferSize, RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst);

        m_Resources->m_ObserveDeviceDataCoherentBindId =
            rhi->RegisterStorageBufferShared(m_Resources->m_ObserveDeviceDataCoherent.get());

        m_Resources->m_MeshCardGPUData.resize(maxAtlasSlots);
        m_Resources->m_MeshCardCoherentGPUData.resize(maxAtlasSlots);
        m_Resources->m_MeshCards.resize(maxAtlasSlots);

        // Then passes configs
        m_Resources->m_SurfaceCachePassBinding = rhi->CreateVertexBufferView();
        m_Resources->m_SurfaceCachePassBinding->AddBinding(
            { 0 }, { RhiImageFormat::RhiImgFmt_R32G32B32_SFLOAT }, { 0 }, 3 * sizeof(float));

        using Ifrit::Runtime::RenderingUtil::CreateComputePassInternal;
        using Ifrit::Runtime::RenderingUtil::CreateGraphicsPassInternal;

        RhiRenderTargetsFormat rtFmt;
        rtFmt.m_colorFormats.push_back(RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM);

        m_Resources->m_SurfaceCachePass =
            CreateGraphicsPassInternal(m_App, Internal::kIntShaderTable.Ayanami.SurfaceCacheGenVS,
                Internal::kIntShaderTable.Ayanami.SurfaceCacheGenFS, 0, 9, rtFmt);

        // RTs
        // TODO: To invalidate the cache, LOAD op is not a good practice?
        m_Resources->m_SurfaceCachePassRTs      = rhi->CreateRenderTargets();
        m_Resources->m_SurfaceCachePassAlbedoRT = rhi->CreateRenderTarget(
            m_Resources->m_SceneCacheAlbdeoAtlas.get(), { 0, 0, 1, 1 }, RhiRenderTargetLoadOp::Load, 0, 0);
        m_Resources->m_SurfaceCachePassNormalRT = rhi->CreateRenderTarget(
            m_Resources->m_SceneCacheNormalAtlas.get(), { 0, 0, 1, 1 }, RhiRenderTargetLoadOp::Load, 0, 0);
        m_Resources->m_SurfaceCachePassDepthRT = rhi->CreateRenderTargetDepthStencil(
            m_Resources->m_SceneCacheTemporaryDepth.get(), { {}, 1.0f }, RhiRenderTargetLoadOp::Clear);

        m_Resources->m_SurfaceCachePassRTs->SetColorAttachments(
            { m_Resources->m_SurfaceCachePassAlbedoRT.get(), m_Resources->m_SurfaceCachePassNormalRT.get() });
        m_Resources->m_SurfaceCachePassRTs->SetDepthStencilAttachment(m_Resources->m_SurfaceCachePassDepthRT.get());

        m_Resources->m_SurfaceCachePass->SetRenderTargetFormat(m_Resources->m_SurfaceCachePassRTs->GetFormat());
        m_Resources->m_SurfaceCachePassRTs->SetRenderArea({ 0, 0, m_Resolution, m_Resolution });
        m_Resources->m_Inited = true;

        // Radiance Cache Pass Related
        m_Resources->m_RadianceCachePass =
            CreateComputePassInternal(m_App, Internal::kIntShaderTable.Ayanami.DirectRadianceInjectionCS, 0, 9);
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateRadianceCacheAtlas(
        const Graphics::Rhi::RhiCommandList* cmdList, Scene* scene)
    {
        auto rhi        = m_App->GetRhi();
        auto numCards   = m_Resources->m_MeshCardIndex.load();
        auto cardGroups = DivRoundUp(numCards, Config::kAyanamiRadianceInjectionObjectsPerBlock);
        auto tileGroups =
            DivRoundUp(m_Resources->m_AtlasElementSize, Config::kAyanamiRadianceInjectionCardSizePerBlock);

        auto perframe = scene->GetPerFrameData();

        m_Resources->m_RadianceCachePass->SetRecordFunction([&](const RhiRenderPassContext* ctx) {
            struct PushConst
            {
                u32 totalCards;
                u32 cardResolution;
                u32 packedShadowMarkBits;
                u32 totalLights;
                u32 atlasResoultion;

                u32 lightDataId;
                u32 radianceOutId;
                u32 cardDataId;
                u32 depthAtlasSRVId;
            } pc;

            pc.totalCards           = numCards;
            pc.cardResolution       = m_Resolution;
            pc.packedShadowMarkBits = m_Resources->m_MaxPerTileLights;
            pc.totalLights          = perframe->m_shadowData2.m_enabledShadowMaps;
            pc.atlasResoultion      = m_Resolution;

            pc.lightDataId     = perframe->m_shadowData2.m_allShadowDataId->GetActiveId();
            pc.radianceOutId   = m_Resources->m_SceneCacheRadianceAtlas->GetDescId();
            pc.cardDataId      = m_Resources->m_ObserveDeviceData->GetDescId();
            pc.depthAtlasSRVId = m_Resources->m_SceneCacheDepthSRV->GetActiveId();

            cmdList->SetPushConst(m_Resources->m_RadianceCachePass, 0, sizeof(PushConst), &pc);
            cmdList->Dispatch(tileGroups, tileGroups, cardGroups);
        });
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateSurfaceModelMatrix()
    {
        for (int i = 0; i < m_Resources->m_MeshCardIndex; i++)
        {
            auto obj                                                = m_Resources->m_MeshCards[i].m_Object;
            auto transform                                          = obj->GetComponent<Transform>();
            auto transformId                                        = transform->GetActiveResourceId();
            m_Resources->m_MeshCardCoherentGPUData[i].m_TransformId = transformId;
        }
    }

    IFRIT_APIDECL Graphics::Rhi::RhiTextureRef AyanamiTrivialSurfaceCacheManager::GetAlbedoAtlas()
    {
        return m_Resources->m_SceneCacheAlbdeoAtlas;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiTextureRef AyanamiTrivialSurfaceCacheManager::GetNormalAtlas()
    {
        return m_Resources->m_SceneCacheNormalAtlas;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiTextureRef AyanamiTrivialSurfaceCacheManager::GetDepthAtlas()
    {
        return m_Resources->m_SceneCacheTemporaryDepth;
    }
} // namespace Ifrit::Runtime::Ayanami