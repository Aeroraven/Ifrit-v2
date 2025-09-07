
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
#include "ifrit/runtime/renderer/ayanami/AyanamiMeshDF.h"
#include "ifrit/runtime/base/MeshComponent.h"
#include "ifrit/runtime/base/Transform.h"
#include "ifrit/runtime/material/SyaroDefaultGBufEmitter.h"
#include "ifrit/runtime/renderer/util/RenderingUtils.h"

#include "ifrit.shader/Ayanami/Ayanami.SharedConst.h"

#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"
#include "ifrit/runtime/rendercore/framegraph/FrameGraphUtils.h"

#include "ifrit/core/math/sampling/LowDiscrepancy.h"

using namespace Ifrit::RHI;
using Ifrit::Math::DivRoundUp;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime::Ayanami
{
    static TConsoleVariable<u32> cvLowDiscrepancySeqLen("cv.Ayanami.SurfaceCache.LowDiscrepancySeqLen", 16,
        "Length of low discrepancy sequence for surface cache generation", CVF_Default);
    static TConsoleVariable<u32> cvSurfaceCacheTemporalAccumMaxHistory(
        "cv.Ayanami.SurfaceCache.TemporalAccumMaxHistory", 32,
        "Maximum history length for temporal accumulation in surface cache", CVF_Default);
    static TConsoleVariable<u32> cvSurfaceCacheResolution(
        "cv.Ayanami.SurfaceCache.Resolution", 4096, "Resolution of the surface cache atlas", CVF_Default);

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

        // For bistro surroundings
        bool         m_ReversedView = false;
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
        using GPUBindId = RHI::RhiDescHandleLegacy;

        bool                                m_Inited             = false;
        bool                                m_RequireGpuDataSync = false;

        RhiTextureRef                       m_SceneCacheAlbdeoAtlas;
        RhiTextureRef                       m_SceneCacheNormalAtlas;
        RhiTextureRef                       m_SceneCacheEmissionAtlas;
        RhiTextureRef                       m_SceneCacheSpecularAtlas;
        RhiTextureRef                       m_SceneCacheTemporaryDepth;
        RhiTextureRef                       m_SceneShadowVisibilityAtlas;
        RhiTextureRef                       m_SceneCacheDirectLightingAtlas;
        RhiTextureRef                       m_SceneCacheIndirrectRadianceAtlas;
        RhiTextureRef                       m_SceneCacheFinalLightingAtlas;

        RhiTextureRef                       m_SceneCacheRadiosityTraceResult;
        RhiTextureRef                       m_SceneCacheRadiositySH_R;
        RhiTextureRef                       m_SceneCacheRadiositySH_G;
        RhiTextureRef                       m_SceneCacheRadiositySH_B;

        // This marks whether a texel (thread group) on surface cache should use
        // offline shadow map or not.
        RhiBufferRef                        m_ShadowMaskOfflineBuffer;

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

        // Debug Controls
        bool                                m_ForceSurfaceCacheRegeneration = false;

        // Frame Graph Resources
        FGTextureNodeRef                    m_RDGSceneCacheAlbedoAtlas;
        FGTextureNodeRef                    m_RDGSceneCacheNormalAtlas;
        FGTextureNodeRef                    m_RDGSceneCacheEmissionAtlas;
        FGTextureNodeRef                    m_RDGSceneCacheSpecularAtlas;
        FGTextureNodeRef                    m_RDGSceneShadowVisibilityAtlas;
        FGTextureNodeRef                    m_RDGSceneCacheDirectLightingAtlas;
        FGTextureNodeRef                    m_RDGSceneCacheIndirectRadianceAtlas;
        FGTextureNodeRef                    m_RDGSceneCacheFinalLightingAtlas;

        FGTextureNodeRef                    m_RDGSceneCacheRadiosityTraceResult;
        FGTextureNodeRef                    m_RDGSceneCacheTemporaryDepth;

        FGTextureNodeRef                    m_RDGSceneCacheRadiositySH_R;
        FGTextureNodeRef                    m_RDGSceneCacheRadiositySH_G;
        FGTextureNodeRef                    m_RDGSceneCacheRadiositySH_B;

        // Low Discrepancy Sequence
        u32                                 m_AccumHistoryLength;
        u32                                 m_ProbeJitterSeqLen;
        Vector2f                            m_ProbeJitter;
    };

    AyanamiTrivialSurfaceCacheManager::AyanamiTrivialSurfaceCacheManager(
        const AyanamiRenderConfig& config, AyanamiSharedContext* sharedCtx, IApplication* app)
        : m_App(app), m_Resolution(cvSurfaceCacheResolution.GetValue())
    {
        m_Resources                                  = new AyanamiTrivialSurfaceCacheManagerResource();
        m_Resources->m_ForceSurfaceCacheRegeneration = config.m_DebugForceSurfaceCacheRegen;
        m_SharedContext                              = sharedCtx;

        m_Resources->m_MaxPerTileLights   = config.m_RadiancePassMaxPerTileLights;
        m_Resources->m_ProbeJitterSeqLen  = cvLowDiscrepancySeqLen.GetValue();
        m_Resources->m_AccumHistoryLength = cvSurfaceCacheTemporalAccumMaxHistory.GetValue();
        PrepareImmutableResource();
    }
    AyanamiTrivialSurfaceCacheManager::~AyanamiTrivialSurfaceCacheManager() { delete m_Resources; }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateSceneCache(Scene* scene)
    {
        using namespace Ifrit::Math;

        m_Resources->m_MeshCardTasks.clear();
        auto objects =
            scene->FilterObjects([](GameObject* obj) { return obj->GetComponent<AyanamiMeshMarker>() != nullptr; });

        for (auto obj : objects)
        {
            auto marker       = obj->GetComponent<AyanamiMeshMarker>();
            auto meshFilter   = obj->GetComponent<MeshFilter>();
            auto meshRenderer = obj->GetComponent<MeshRenderer>();
            auto meshdf       = obj->GetComponent<AyanamiMeshDF>();

            if (meshFilter == nullptr)
            {
                IF_LOG_CRITICAL("Ayanami.SurfaceCache", "MeshFilter is nullptr");
            }

            if (meshRenderer == nullptr)
            {
                IF_LOG_CRITICAL("Ayanami.SurfaceCache", "MeshRenderer is nullptr");
            }

            auto material = meshRenderer->GetMaterial();
            if (material == nullptr)
            {
                IF_LOG_CRITICAL("Ayanami.SurfaceCache", "Material is nullptr");
            }

            // TODO: Batcher should be used to batch the meshes with same material
            // for simplicity, we just use the default material
            using Ifrit::CheckedPointerCast;
            auto castedMaterial = ForcedCheckedCast<SyaroDefaultGBufEmitter>(material);
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
                auto indexCounts   = SizeCast<u32>(meshData->m_indices.size());

                if (vertexBuffer == nullptr || indexBuffer == nullptr)
                {
                    IF_LOG_CRITICAL("Ayanami.SurfaceCache", "Vertex buffer or index buffer is nullptr");
                }

                auto            objectBufferId = meshResource.objectBuffer->GetDescId();
                auto            meshBBoxMax    = meshData->m_BoundingBoxMax;
                auto            meshBBoxMin    = meshData->m_BoundingBoxMin;
                auto            meshBBoxCenter = (meshBBoxMax + meshBBoxMin) * 0.5f;
                auto            meshBBoxSize   = meshBBoxMax - meshBBoxMin;
                auto            meshBBoxExtent = meshBBoxSize * 0.5f;

                auto&           meshVertices = meshData->m_verticesAligned;

                Array<f32, 3>   meshBBoxExtentArr = { meshBBoxExtent.x, meshBBoxExtent.y, meshBBoxExtent.z };
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
                    f32      viewNearPlane             = 10.0f;
                    f32      viewNearPlaneCompensation = 1.0f;
                    f32      cardZCompensation         = 10.0f;
                    Vector3f viewLocation              = meshBBoxCenter - card.m_CardDirection * cardExtent.z
                        - card.m_CardDirection * (viewNearPlane + viewNearPlaneCompensation);

                    Vector3f   viewUp     = kCardLookAtUps[i];
                    Vector3f   viewTarget = meshBBoxCenter;
                    Matrix4x4f viewMatrix = LookAt(viewLocation, viewTarget, viewUp);

                    f32        viewAspect = cardExtent.x / cardExtent.y;
                    Matrix4x4f viewOrtho  = OrthographicNegateY(cardExtent.y * 2.0f, viewAspect, viewNearPlane,
                         cardExtent.z * 2.0f + viewNearPlane + cardZCompensation + viewNearPlaneCompensation);

                    // printf("Card %d:\n", slotId);
                    // printf("ViewExtent: %f, %f, %f\n", cardExtent.x, cardExtent.y, cardExtent.z);
                    // printf("ViewCenter: %f, %f, %f\n", meshBBoxCenter.x, meshBBoxCenter.y, meshBBoxCenter.z);
                    // printf("ViewLocation: %f, %f, %f\n", viewLocation.x, viewLocation.y, viewLocation.z);

                    Matrix4x4f viewProj = viewOrtho;
                    Matrix4x4f viewVP   = MatMul(viewProj, viewMatrix);

                    card.m_ObserverView = viewMatrix;
                    card.m_ObserverProj = viewOrtho;
                    card.m_ObserverVP   = viewVP;

                    card.m_TempAlbedoId = albedoId;
                    card.m_TempNormalId = normalId;

                    if (meshdf->IsDoubleSided())
                    {
                        card.m_ReversedView = true;
                    }

                    m_Resources->m_MeshCardGPUData[slotId].m_ObserverVP        = Transpose(viewVP);
                    m_Resources->m_MeshCardGPUData[slotId].m_ObserverVPInverse = Transpose(Inverse(viewVP));

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

            auto tq           = m_App->GetRhi()->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
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

    IFRIT_APIDECL GraphicsPassNode& AyanamiTrivialSurfaceCacheManager::UpdateSurfaceCacheAtlas(
        FrameGraphBuilder& builder)
    {

        auto& pass = builder.AddGraphicsPass("Ayanami/SurfaceCacheGenPass",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.SurfaceCacheGenVS, {}),
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.SurfaceCacheGenFS, {}), 9);

        pass.SetExecutionFunction([this](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;

            for (auto id : m_Resources->m_MeshCardTasks)
            {
                RhiViewport      viewport;
                ManagedMeshCard& card = m_Resources->m_MeshCards[id];
                viewport.x            = 1.0f * card.m_CardLocation.x;
                viewport.y            = 1.0f * card.m_CardLocation.y;
                viewport.width        = 1.0f * card.m_CardExtent.x;
                viewport.height       = 1.0f * card.m_CardExtent.y;
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

                if (card.m_ReversedView)
                {
                    cmd->SetCullMode(RhiCullMode::Back);
                }
                else
                {
                    cmd->SetCullMode(RhiCullMode::None);
                }

                auto vioPass = const_cast<RHI::RhiGraphicsPass*>(ctx.m_GraphicsPass);
                cmd->SetPushConst(&pc, 0, sizeof(PushConst));
                cmd->DrawIndexed(card.m_IndexCounts, 1, 0, 0, 0);
            }

            if (!m_Resources->m_ForceSurfaceCacheRegeneration)
            {
                m_Resources->m_MeshCardTasks.clear();
            }
        });
        pass.AddRenderTarget(*m_Resources->m_RDGSceneCacheAlbedoAtlas)
            .AddRenderTarget(*m_Resources->m_RDGSceneCacheNormalAtlas)
            .AddDepthTarget(*m_Resources->m_RDGSceneCacheTemporaryDepth);

        return pass;
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::InitContext(FrameGraphBuilder& builder)
    {
        m_Resources->m_RDGSceneCacheAlbedoAtlas =
            &builder.ImportTexture("Ayanami.SceneCacheAlbedoAtlas", m_Resources->m_SceneCacheAlbdeoAtlas.get());
        m_Resources->m_RDGSceneCacheNormalAtlas =
            &builder.ImportTexture("Ayanami.SceneCacheNormalAtlas", m_Resources->m_SceneCacheNormalAtlas.get());
        m_Resources->m_RDGSceneCacheEmissionAtlas =
            &builder.ImportTexture("Ayanami.SceneCacheEmissionAtlas", m_Resources->m_SceneCacheEmissionAtlas.get());
        m_Resources->m_RDGSceneCacheSpecularAtlas =
            &builder.ImportTexture("Ayanami.SceneCacheSpecularAtlas", m_Resources->m_SceneCacheSpecularAtlas.get());
        m_Resources->m_RDGSceneShadowVisibilityAtlas = &builder.ImportTexture(
            "Ayanami.SceneShadowVisibilityAtlas", m_Resources->m_SceneShadowVisibilityAtlas.get());
        m_Resources->m_RDGSceneCacheDirectLightingAtlas = &builder.ImportTexture(
            "Ayanami.SceneCacheDirectLightingAtlas", m_Resources->m_SceneCacheDirectLightingAtlas.get());
        m_Resources->m_RDGSceneCacheIndirectRadianceAtlas = &builder.ImportTexture(
            "Ayanami.SceneCacheIndirectRadianceAtlas", m_Resources->m_SceneCacheIndirrectRadianceAtlas.get());
        m_Resources->m_RDGSceneCacheFinalLightingAtlas = &builder.ImportTexture(
            "Ayanami.SceneCacheFinalLightingAtlas", m_Resources->m_SceneCacheFinalLightingAtlas.get());

        m_Resources->m_RDGSceneCacheTemporaryDepth =
            &builder.ImportTexture("Ayanami.SceneCacheTemporaryDepth", m_Resources->m_SceneCacheTemporaryDepth.get());
        m_Resources->m_RDGSceneCacheRadiosityTraceResult = &builder.ImportTexture(
            "Ayanami.SceneCacheRadiosityTraceResult", m_Resources->m_SceneCacheRadiosityTraceResult.get());
        m_Resources->m_RDGSceneCacheRadiositySH_R =
            &builder.ImportTexture("Ayanami.SceneCacheRadiositySH_R", m_Resources->m_SceneCacheRadiositySH_R.get());
        m_Resources->m_RDGSceneCacheRadiositySH_G =
            &builder.ImportTexture("Ayanami.SceneCacheRadiositySH_G", m_Resources->m_SceneCacheRadiositySH_G.get());
        m_Resources->m_RDGSceneCacheRadiositySH_B =
            &builder.ImportTexture("Ayanami.SceneCacheRadiositySH_B", m_Resources->m_SceneCacheRadiositySH_B.get());
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::PrepareImmutableResource()
    {
        if (m_Resources->m_Inited)
            return;
        auto rhi                 = m_App->GetRhi();
        auto linearRepeatSampler = m_App->GetSharedRenderResource()->GetLinearRepeatSampler();

        m_Resources->m_SceneCacheAlbdeoAtlas      = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_AlbedoAtlas",
                 m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8G8B8A8_UNORM,
                 RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_RenderTarget, false);
        m_Resources->m_SceneCacheNormalAtlas      = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_NormalAtlas",
                 m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                 RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_RenderTarget, false);
        m_Resources->m_SceneCacheEmissionAtlas    = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_EmissionAtlas",
               m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8_UNORM,
               RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_RenderTarget, false);
        m_Resources->m_SceneCacheSpecularAtlas    = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_SpecularAtlas",
               m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8_UNORM,
               RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_RenderTarget, false);
        m_Resources->m_SceneShadowVisibilityAtlas = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_RadianceAtlas",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8G8_UNORM,
            RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess
                | RhiImageUsage::RhiImgUsage_RenderTarget,
            true);
        m_Resources->m_SceneCacheDirectLightingAtlas =
            rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_DirectLightingAtlas", m_Resolution, m_Resolution,
                RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess, true);
        m_Resources->m_SceneCacheIndirrectRadianceAtlas =
            rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_IndirectRadianceAtlas", m_Resolution, m_Resolution,
                RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess
                    | RhiImageUsage::RhiImgUsage_RenderTarget,
                true);
        m_Resources->m_SceneCacheFinalLightingAtlas =
            rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_FinalLightingAtlas", m_Resolution, m_Resolution,
                RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess
                    | RhiImageUsage::RhiImgUsage_RenderTarget | RhiImageUsage::RhiImgUsage_CopyDst,
                true);

        m_Resources->m_SceneCacheRadiosityTraceResult =
            rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_RadiosityTraceResult", m_Resolution, m_Resolution,
                RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess, true);
        m_Resources->m_SceneCacheRadiositySH_R = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_RadiositySH_R",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
            RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess, true);
        m_Resources->m_SceneCacheRadiositySH_G = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_RadiositySH_G",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
            RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess, true);
        m_Resources->m_SceneCacheRadiositySH_B = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_RadiositySH_B",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
            RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess, true);

        // A depth buffer is required for the surface cache pass
        m_Resources->m_SceneCacheTemporaryDepth =
            rhi->CreateDepthTexture("AyanamiTrivialSurfaceCache_DepthAtlas", m_Resolution, m_Resolution, false);

        auto maxAtlasSlots =
            m_Resolution * m_Resolution / m_Resources->m_AtlasElementSize / m_Resources->m_AtlasElementSize;
        auto requiredObserverBufferSize = SizeCast<u32>(maxAtlasSlots * sizeof(ManagedMeshCardGPUData));
        auto requiredCoherentBufferSize = SizeCast<u32>(maxAtlasSlots * sizeof(ManagedMeshCardCoherentGPUData));

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
    }

    IFRIT_APIDECL ComputePassNode& AyanamiTrivialSurfaceCacheManager::UpdateShadowVisibilityAtlas(
        FrameGraphBuilder& builder, Scene* scene)
    {
        auto rhi        = m_App->GetRhi();
        auto numCards   = m_Resources->m_MeshCardIndex.load();
        auto cardGroups = DivRoundUp(numCards, Config::kAyanamiShadowVisibilityObjectsPerBlock);
        auto tileGroups = DivRoundUp(m_Resources->m_AtlasElementSize, Config::kAyanamiShadowVisibilityCardSizePerBlock);
        auto perframe   = scene->GetPerFrameData();
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

            u32 worldObjTransforms;
            u32 perframeId;
            u32 m_NormalAtlasSRV;
        } pc;
        pc.totalCards           = numCards;
        pc.cardResolution       = m_Resources->m_AtlasElementSize;
        pc.packedShadowMarkBits = m_Resources->m_MaxPerTileLights;
        pc.totalLights          = perframe->m_shadowData2.m_enabledShadowMaps;
        pc.atlasResoultion      = m_Resolution;

        pc.lightDataId        = perframe->m_shadowData2.m_allShadowDataId->GetActiveId();
        pc.radianceOutId      = 0;
        pc.cardDataId         = m_Resources->m_ObserveDeviceData->GetDescId();
        pc.depthAtlasSRVId    = 0;
        pc.worldObjTransforms = m_Resources->m_ObserveDeviceDataCoherentBindId->GetActiveId();
        pc.perframeId         = perframe->m_views[0].m_viewBufferId->GetActiveId();

        pc.m_NormalAtlasSRV = 0;

        UpdateSurfaceModelMatrix();
        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.SurfaceCache.CameraShadowVisibility",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.DirectShadowVisibilityCS, {}),
            Vector3i{ (i32)tileGroups, (i32)tileGroups, (i32)cardGroups }, pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_NormalAtlasSRV = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);
                data.depthAtlasSRVId  = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheTemporaryDepth);
                data.radianceOutId    = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneShadowVisibilityAtlas);
                SetRootConstant(data, ctx);
            });
        pass.AddWriteResource(*m_Resources->m_RDGSceneShadowVisibilityAtlas)
            .AddReadResource(*m_Resources->m_RDGSceneCacheTemporaryDepth);

        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AyanamiTrivialSurfaceCacheManager::UpdateRadiosityTrace(FrameGraphBuilder& builder,
        Scene* scene, FGTextureNodeRef globalDFSRV, FGBufferNodeRef objectGridsUAV, u32 meshDFList,
        Vector3f globalDFMin, Vector3f globalDFMax, u32 globalDFResolution, u32 voxelsPerGdfWidth)
    {
        // pass 0: clear final lightint atlas on frame 1
        if (m_SharedContext->m_FrameIdx == 1)
        {
            AddClearUAVTexturePass(builder, "Ayanami.Radiosity.ClearLightingAtlas",
                *m_Resources->m_RDGSceneCacheFinalLightingAtlas, Vector4f(0.0f, 0.0f, 0.0f, 1.0f));
        }

        // pass 1: radiosity trace
        struct PushConst
        {
            Vector4f m_GlobalDFBoxMin;
            Vector4f m_GlobalDFBoxMax;
            Vector2f m_TraceCoordJitter;
            Vector2f m_ProbeCenterJitter;
            u32      m_TraceRadianceAtlasUAV;
            u32      m_GlobalDFSRV;
            u32      m_CardResolution;
            u32      m_CardAtlasResolution;
            u32      m_CardDepthAtlasSRV;
            u32      m_CardNormalAtlasSRV;
            u32      m_CardLightingAtlasSRV;
            u32      m_AllCardObjDataId;
            u32      m_AllMeshDFDataId;
            u32      m_NumTotalCards;
            u32      m_GlobalDFResolution;
            u32      m_VoxelsPerWidth;
            u32      m_ObjectGridUAV;
        } pc;

        pc.m_GlobalDFBoxMin        = Vector4f(globalDFMin, 0.0f);
        pc.m_GlobalDFBoxMax        = Vector4f(globalDFMax, 0.0f);
        pc.m_TraceCoordJitter      = m_Resources->m_ProbeJitter;
        pc.m_ProbeCenterJitter     = m_Resources->m_ProbeJitter;
        pc.m_TraceRadianceAtlasUAV = 0;
        pc.m_GlobalDFSRV           = 0;
        pc.m_CardResolution        = m_Resources->m_AtlasElementSize;
        pc.m_CardAtlasResolution   = m_Resolution;
        pc.m_CardDepthAtlasSRV     = 0;
        pc.m_CardNormalAtlasSRV    = 0;
        pc.m_CardLightingAtlasSRV  = 0;
        pc.m_AllCardObjDataId      = m_Resources->m_ObserveDeviceData->GetDescId();
        pc.m_AllMeshDFDataId       = meshDFList;
        pc.m_NumTotalCards         = m_Resources->m_MeshCardIndex.load();
        pc.m_GlobalDFResolution    = globalDFResolution;
        pc.m_VoxelsPerWidth        = voxelsPerGdfWidth;
        pc.m_ObjectGridUAV         = 0;

        auto totalCardTiles = m_Resources->m_AtlasElementSize * m_Resources->m_AtlasElementSize
            / (Config::kAyanami_CardTileWidth * Config::kAyanami_CardTileWidth) * pc.m_NumTotalCards;
        auto  totalTraces = totalCardTiles * Config::kAyanami_RadiosityTracesPerCardTile;
        // iDebug("Total Traces: {}", totalTraces);
        // iDebug("Total Card Tiles: {}", totalCardTiles);
        auto  numTGs = DivRoundUp<i32, i32>(totalTraces, Config::kAyanamiRadiosityTraceKernelSize);

        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.Radiosity.Trace",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.RadiosityTraceCS, {}), Vector3i{ numTGs, 1, 1 }, pc,
            [globalDFSRV, objectGridsUAV, this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_GlobalDFSRV        = ctx.m_FgDesc->GetSRV(*globalDFSRV);
                data.m_CardDepthAtlasSRV  = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheTemporaryDepth);
                data.m_CardNormalAtlasSRV = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);
                // data.m_CardLightingAtlasSRV  =
                // ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheDirectLightingAtlas);
                data.m_CardLightingAtlasSRV  = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheFinalLightingAtlas);
                data.m_TraceRadianceAtlasUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiosityTraceResult);
                data.m_ObjectGridUAV         = ctx.m_FgDesc->GetUAV(*objectGridsUAV);
                SetRootConstant(data, ctx);
            });

        pass.AddWriteResource(*m_Resources->m_RDGSceneCacheRadiosityTraceResult)
            .AddReadResource(*m_Resources->m_RDGSceneCacheNormalAtlas)
            .AddReadResource(*m_Resources->m_RDGSceneCacheTemporaryDepth)
            .AddReadResource(*m_Resources->m_RDGSceneCacheDirectLightingAtlas)
            .AddReadResource(*m_Resources->m_RDGSceneCacheFinalLightingAtlas)
            .AddReadResource(*objectGridsUAV)
            .AddReadResource(*globalDFSRV);

        return pass;
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::RadiositySHConversion(
        FrameGraphBuilder& builder, u32 meshDFList)
    {
        // TODO
        struct PushConst
        {
            Vector2f m_TraceCoordJitter;
            Vector2f m_ProbeCenterJitter;
            u32      m_CardAtlasResolution;
            u32      m_CardResolution;
            u32      m_NumTotalCards;
            u32      m_CardDepthAtlasSRV;
            u32      m_CardNormalAtlasSRV;
            u32      m_AllCardObjDataId;
            u32      m_AllMeshDFDataId;
            u32      m_FilteredRadianceAtlasUAV;
            u32      m_RWRadiosityProbeSHAtlasRUAV;
            u32      m_RWRadiosityProbeSHAtlasGUAV;
            u32      m_RWRadiosityProbeSHAtlasBUAV;
            u32      m_TotalProbes;
        } pc;
        pc.m_TraceCoordJitter            = m_Resources->m_ProbeJitter;
        pc.m_ProbeCenterJitter           = m_Resources->m_ProbeJitter;
        pc.m_CardAtlasResolution         = m_Resolution;
        pc.m_CardResolution              = m_Resources->m_AtlasElementSize;
        pc.m_NumTotalCards               = m_Resources->m_MeshCardIndex.load();
        pc.m_CardDepthAtlasSRV           = 0;
        pc.m_CardNormalAtlasSRV          = 0;
        pc.m_AllCardObjDataId            = m_Resources->m_ObserveDeviceData->GetDescId();
        pc.m_AllMeshDFDataId             = meshDFList;
        pc.m_FilteredRadianceAtlasUAV    = 0;
        pc.m_RWRadiosityProbeSHAtlasRUAV = 0;
        pc.m_RWRadiosityProbeSHAtlasGUAV = 0;
        pc.m_RWRadiosityProbeSHAtlasBUAV = 0;

        auto numCardTiles = m_Resources->m_AtlasElementSize * m_Resources->m_AtlasElementSize
            / (Config::kAyanami_CardTileWidth * Config::kAyanami_CardTileWidth) * pc.m_NumTotalCards;
        auto numProbes = numCardTiles
            * (Config::kAyanami_RadiosityProbesPerCardTileWidth * Config::kAyanami_RadiosityProbesPerCardTileWidth);

        auto numTGs      = DivRoundUp<i32, i32>(numProbes, Config::kAyanamiSphericalHarmonicsCvtKernelSize);
        pc.m_TotalProbes = numProbes;

        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.Radiosity.SHConversion",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.RadiositySHConversionCS, {}), Vector3i{ numTGs, 1, 1 },
            pc, [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_CardDepthAtlasSRV  = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheTemporaryDepth);
                data.m_CardNormalAtlasSRV = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);

                // TODO: this is not filtered
                data.m_FilteredRadianceAtlasUAV =
                    ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiosityTraceResult);
                data.m_RWRadiosityProbeSHAtlasRUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiositySH_R);
                data.m_RWRadiosityProbeSHAtlasGUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiositySH_G);
                data.m_RWRadiosityProbeSHAtlasBUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiositySH_B);
                SetRootConstant(data, ctx);
            });

        pass.AddWriteResource(*m_Resources->m_RDGSceneCacheRadiositySH_R)
            .AddWriteResource(*m_Resources->m_RDGSceneCacheRadiositySH_G)
            .AddWriteResource(*m_Resources->m_RDGSceneCacheRadiositySH_B)
            .AddReadResource(*m_Resources->m_RDGSceneCacheTemporaryDepth)
            .AddReadResource(*m_Resources->m_RDGSceneCacheNormalAtlas)
            .AddReadResource(*m_Resources->m_RDGSceneCacheRadiosityTraceResult);
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::RadiositySHIntegrate(
        FrameGraphBuilder& builder, u32 meshDFList)
    {
        struct PushConst
        {
            u32 m_CardResolution;
            u32 m_CardAtlasResolution;
            u32 m_MeshDFDescIdUAV;
            u32 m_CardNormalAtlasSRV;
            u32 m_RadiosityProbeSHAtlasRUAV;
            u32 m_RadiosityProbeSHAtlasGUAV;
            u32 m_RadiosityProbeSHAtlasBUAV;
            u32 m_SurfaceIndirectLightingUAV;
        } pc;
        pc.m_CardResolution             = m_Resources->m_AtlasElementSize;
        pc.m_CardAtlasResolution        = m_Resolution;
        pc.m_MeshDFDescIdUAV            = meshDFList;
        pc.m_CardNormalAtlasSRV         = 0;
        pc.m_RadiosityProbeSHAtlasRUAV  = 0;
        pc.m_RadiosityProbeSHAtlasGUAV  = 0;
        pc.m_RadiosityProbeSHAtlasBUAV  = 0;
        pc.m_SurfaceIndirectLightingUAV = 0;

        auto numCards     = m_Resources->m_MeshCardIndex.load();
        auto numCardTiles = m_Resources->m_AtlasElementSize * m_Resources->m_AtlasElementSize
            / (Config::kAyanami_CardTileWidth * Config::kAyanami_CardTileWidth) * numCards;
        auto  numTGs = static_cast<i32>(numCardTiles);

        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.Radiosity.SHIntegrate",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.RadiositySHIntegrateCS, {}), Vector3i{ numTGs, 1, 1 },
            pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_CardNormalAtlasSRV        = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);
                data.m_RadiosityProbeSHAtlasRUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiositySH_R);
                data.m_RadiosityProbeSHAtlasGUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiositySH_G);
                data.m_RadiosityProbeSHAtlasBUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheRadiositySH_B);
                data.m_SurfaceIndirectLightingUAV =
                    ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheIndirectRadianceAtlas);
                SetRootConstant(data, ctx);
            })
                         .AddReadResource(*m_Resources->m_RDGSceneCacheNormalAtlas)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheRadiositySH_R)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheRadiositySH_G)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheRadiositySH_B)
                         .AddWriteResource(*m_Resources->m_RDGSceneCacheIndirectRadianceAtlas);
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateDirectLighting(
        FrameGraphBuilder& builder, u32 meshDFList, Vector3f lightDir)
    {
        struct PushConst
        {
            Vector4f m_LightDir;
            u32      m_DirectLightUAV;
            u32      m_NormalAtlasSRV;
            u32      m_ShadowMaskSRV;
            u32      m_CardResolution;
            u32      m_CardAtlasResolution;
            u32      m_MeshDFDescListId;
        } pc;
        pc.m_DirectLightUAV      = 0;
        pc.m_CardResolution      = m_Resources->m_AtlasElementSize;
        pc.m_CardAtlasResolution = m_Resolution;
        pc.m_MeshDFDescListId    = meshDFList;
        pc.m_ShadowMaskSRV       = 0;
        pc.m_NormalAtlasSRV      = 0;
        pc.m_LightDir            = Vector4f(lightDir, 0.0f);

        auto  numCards   = m_Resources->m_MeshCardIndex.load();
        auto  cardGroups = DivRoundUp(numCards, Config::kAyanamiSCDirectLightObjectsPerBlock);
        auto  tileGroups = DivRoundUp(m_Resources->m_AtlasElementSize, Config::kAyanamiSCDirectLightCardSizePerBlock);

        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.SurfaceCache.DirectLighting",
            ShaderVariantDesc(Internal::kIntShaderTableAyanami.SurfaceCacheDirectLightCS, {}),
            Vector3i{ (i32)tileGroups, (i32)tileGroups, (i32)cardGroups }, pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_DirectLightUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheDirectLightingAtlas);
                data.m_ShadowMaskSRV  = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneShadowVisibilityAtlas);
                data.m_NormalAtlasSRV = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);
                SetRootConstant(data, ctx);
            })
                         .AddWriteResource(*m_Resources->m_RDGSceneCacheDirectLightingAtlas)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheNormalAtlas)
                         .AddReadResource(*m_Resources->m_RDGSceneShadowVisibilityAtlas);
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::UpdateSurfaceModelMatrix()
    {
        for (u32 i = 0; i < m_Resources->m_MeshCardIndex; i++)
        {
            auto obj                                                = m_Resources->m_MeshCards[i].m_Object;
            auto transform                                          = obj->GetComponent<Transform>();
            auto transformId                                        = transform->GetActiveResourceId();
            m_Resources->m_MeshCardCoherentGPUData[i].m_TransformId = transformId;
        }
        // Then update the coherent buffer
        auto tq           = m_App->GetRhi()->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
        auto tgt          = m_Resources->m_ObserveDeviceDataCoherent->GetActiveBuffer();
        auto stagedBuffer = m_App->GetRhi()->CreateStagedSingleBuffer(tgt);
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            stagedBuffer->CmdCopyToDevice(cmd, m_Resources->m_MeshCardCoherentGPUData.data(),
                SizeCast<u32>(m_Resources->m_MeshCardCoherentGPUData.size() * sizeof(ManagedMeshCardCoherentGPUData)),
                0);
        });
    }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::CombineLighting(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_FrameIdx; // clamped to max history !!!
            u32 m_DirectLightingAtlasSRV;
            u32 m_IndirectLightingAtlasSRV;
            u32 m_AlbedoAtlasSRV;
            u32 m_FinalLightingAtlasUAV;
            u32 m_CardResolution;
            u32 m_CardAtlasResolution;
        } pc;
        pc.m_FrameIdx = static_cast<u32>(
            std::min(m_SharedContext->m_FrameIdx, static_cast<u64>(m_Resources->m_AccumHistoryLength)));
        pc.m_DirectLightingAtlasSRV   = 0;
        pc.m_IndirectLightingAtlasSRV = 0;
        pc.m_AlbedoAtlasSRV           = 0;
        pc.m_FinalLightingAtlasUAV    = 0;
        pc.m_CardResolution           = m_Resources->m_AtlasElementSize;
        pc.m_CardAtlasResolution      = m_Resolution;

        auto numCards   = m_Resources->m_MeshCardIndex.load();
        auto cardGroups = DivRoundUp(numCards, Config::kAyanamiSCDirectLightObjectsPerBlock);
        auto tileGroups = DivRoundUp(m_Resources->m_AtlasElementSize, Config::kAyanamiSCDirectLightCardSizePerBlock);
        Vector3i numTGs{ (i32)tileGroups, (i32)tileGroups, (i32)cardGroups };

        auto&    pass = AddComputePass<PushConst>(builder, "Ayanami.SurfaceCache.CombineLighting",
               ShaderVariantDesc(Internal::kIntShaderTableAyanami.SurfaceCacheCombineLightCS, {}), numTGs, pc,
               [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_DirectLightingAtlasSRV = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheDirectLightingAtlas);
                data.m_IndirectLightingAtlasSRV =
                    ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheIndirectRadianceAtlas);
                data.m_AlbedoAtlasSRV        = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheAlbedoAtlas);
                data.m_FinalLightingAtlasUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheFinalLightingAtlas);
                SetRootConstant(data, ctx);
            })
                         .AddWriteResource(*m_Resources->m_RDGSceneCacheFinalLightingAtlas)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheDirectLightingAtlas)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheIndirectRadianceAtlas)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheAlbedoAtlas);
    }

    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGAlbedoAtlas()
    {
        return *m_Resources->m_RDGSceneCacheAlbedoAtlas;
    }

    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGNormalAtlas()
    {
        return *m_Resources->m_RDGSceneCacheNormalAtlas;
    }

    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGDepthAtlas()
    {
        return *m_Resources->m_RDGSceneCacheTemporaryDepth;
    }

    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGShadowVisibilityAtlas()
    {
        return *m_Resources->m_RDGSceneShadowVisibilityAtlas;
    }

    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGTracedRadianceAtlas()
    {
        return *m_Resources->m_RDGSceneCacheIndirectRadianceAtlas;
    }

    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGDirectLightingAtlas()
    {
        return *m_Resources->m_RDGSceneCacheDirectLightingAtlas;
    }

    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGIndirectLightingAtlas()
    {
        return *m_Resources->m_RDGSceneCacheIndirectRadianceAtlas;
    }
    IFRIT_APIDECL FGTextureNode& AyanamiTrivialSurfaceCacheManager::GetRDGFinalLightingAtlas()
    {
        return *m_Resources->m_RDGSceneCacheFinalLightingAtlas;
    }

    IFRIT_APIDECL RHI::RhiBufferRef AyanamiTrivialSurfaceCacheManager::GetCardDataBuffer()
    {
        return m_Resources->m_ObserveDeviceData;
    }

    IFRIT_APIDECL u32 AyanamiTrivialSurfaceCacheManager::GetCardResolution() { return m_Resources->m_AtlasElementSize; }
    IFRIT_APIDECL u32 AyanamiTrivialSurfaceCacheManager::GetCardAtlasResolution() { return m_Resolution; }
    IFRIT_APIDECL u32 AyanamiTrivialSurfaceCacheManager::GetWorldMatsId()
    {
        return m_Resources->m_ObserveDeviceDataCoherentBindId->GetActiveId();
    }
    IFRIT_APIDECL u32  AyanamiTrivialSurfaceCacheManager::GetNumCards() { return m_Resources->m_MeshCardIndex.load(); }

    IFRIT_APIDECL void AyanamiTrivialSurfaceCacheManager::FrameProceed()
    {
        using namespace Math;

        auto frameIdx              = m_SharedContext->m_FrameIdx;
        auto seqEleId              = static_cast<u32>(frameIdx % m_Resources->m_ProbeJitterSeqLen);
        auto jitter                = Math::Hammersley2d(seqEleId, m_Resources->m_ProbeJitterSeqLen);
        m_Resources->m_ProbeJitter = jitter - Vector2f(0.5f);
    }
} // namespace Ifrit::Runtime::Ayanami