
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

#include "ifrit/runtime/renderer/internal/InternalShaderRegistry.Ayanami.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"

using namespace Ifrit::Graphics::Rhi;
using Ifrit::Math::DivRoundUp;
using namespace Ifrit::Runtime::FrameGraphUtils;

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

        RhiTextureRef                       m_SceneCacheAlbdeoAtlas;
        RhiTextureRef                       m_SceneCacheNormalAtlas;
        RhiTextureRef                       m_SceneCacheEmissionAtlas;
        RhiTextureRef                       m_SceneCacheSpecularAtlas;
        RhiTextureRef                       m_SceneCacheTemporaryDepth;
        RhiTextureRef                       m_SceneShadowVisibilityAtlas;
        RhiTextureRef                       m_SceneDirectLightingAtlas;
        RhiTextureRef                       m_SceneCacheIndirrectRadianceAtlas;

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
        FGTextureNodeRef                    m_RDGSceneDirectLighting;
        FGTextureNodeRef                    m_RDGSceneCacheIndirectRadianceAtlas;
        FGTextureNodeRef                    m_RDGSceneCacheTemporaryDepth;
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
                    f32      viewNearPlane = 0.1f;
                    Vector3f viewLocation =
                        meshBBoxCenter - card.m_CardDirection * cardExtent.z - card.m_CardDirection * viewNearPlane;

                    Vector3f   viewUp     = kCardLookAtUps[i];
                    Vector3f   viewTarget = meshBBoxCenter;
                    Matrix4x4f viewMatrix = LookAt(viewLocation, viewTarget, viewUp);

                    f32        viewAspect = cardExtent.x / cardExtent.y;
                    Matrix4x4f viewOrtho  = OrthographicNegateY(
                        cardExtent.y * 2.0, viewAspect, viewNearPlane, cardExtent.z * 2.0f + viewNearPlane);

                    // printf("ViewExtent: %f, %f, %f\n", cardExtent.x, cardExtent.y, cardExtent.z);
                    // printf("ViewCenter: %f, %f, %f\n", meshBBoxCenter.x, meshBBoxCenter.y, meshBBoxCenter.z);

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
            Internal::kIntShaderTableAyanami.SurfaceCacheGenVS, Internal::kIntShaderTableAyanami.SurfaceCacheGenFS, 9);

        pass.SetExecutionFunction([this](const FrameGraphPassContext& ctx) {
            auto cmd = ctx.m_CmdList;

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

                auto vioPass = const_cast<Graphics::Rhi::RhiGraphicsPass*>(ctx.m_GraphicsPass);
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
        m_Resources->m_RDGSceneDirectLighting =
            &builder.ImportTexture("Ayanami.SceneDirectLightingAtlas", m_Resources->m_SceneDirectLightingAtlas.get());
        m_Resources->m_RDGSceneCacheIndirectRadianceAtlas = &builder.ImportTexture(
            "Ayanami.SceneCacheIndirectRadianceAtlas", m_Resources->m_SceneCacheIndirrectRadianceAtlas.get());
        m_Resources->m_RDGSceneCacheTemporaryDepth =
            &builder.ImportTexture("Ayanami.SceneCacheTemporaryDepth", m_Resources->m_SceneCacheTemporaryDepth.get());
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
                 m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R8G8_SNORM,
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
        m_Resources->m_SceneDirectLightingAtlas = rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_DirectLightingAtlas",
            m_Resolution, m_Resolution, RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
            RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess, true);
        m_Resources->m_SceneCacheIndirrectRadianceAtlas =
            rhi->CreateTexture2D("AyanamiTrivialSurfaceCache_IndirectRadianceAtlas", m_Resolution, m_Resolution,
                RhiImageFormat::RhiImgFmt_R16G16B16A16_SFLOAT,
                RhiImageUsage::RhiImgUsage_ShaderRead | RhiImageUsage::RhiImgUsage_UnorderedAccess
                    | RhiImageUsage::RhiImgUsage_RenderTarget,
                true);

        // A depth buffer is required for the surface cache pass
        m_Resources->m_SceneCacheTemporaryDepth =
            rhi->CreateDepthTexture("AyanamiTrivialSurfaceCache_DepthAtlas", m_Resolution, m_Resolution, false);

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
        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.CameraShadowVisibilityPass",
            Internal::kIntShaderTableAyanami.DirectShadowVisibilityCS,
            Vector3i{ (i32)tileGroups, (i32)tileGroups, (i32)cardGroups }, pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_NormalAtlasSRV = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);
                data.depthAtlasSRVId  = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheTemporaryDepth);
                data.radianceOutId    = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneShadowVisibilityAtlas);
                SetRootSignature(data, ctx);
            });
        pass.AddWriteResource(*m_Resources->m_RDGSceneShadowVisibilityAtlas)
            .AddReadResource(*m_Resources->m_RDGSceneCacheTemporaryDepth);

        return pass;
    }

    IFRIT_APIDECL ComputePassNode& AyanamiTrivialSurfaceCacheManager::UpdateIndirectRadianceCacheAtlas(
        FrameGraphBuilder& builder, Scene* scene, FGTextureNodeRef globalDFSRV, u32 meshDFList)
    {
        struct PushConst
        {
            Vector2f m_TraceCoordJitter;
            Vector2f m_ProbeCenterJitter;
            u32      m_TraceRadianceAtlasUAV;
            u32      m_GlobalDFSRV;
            u32      m_CardResolution;
            u32      m_CardAtlasResolution;
            u32      m_CardDepthAtlasSRV;
            u32      m_CardNormalAtlasSRV;
            u32      m_AllCardObjDataId;
            u32      m_AllMeshDFDataId;
            u32      m_NumTotalCards;
        } pc;

        pc.m_TraceCoordJitter      = Vector2f(0.0f, 0.0f);
        pc.m_ProbeCenterJitter     = Vector2f(0.0f, 0.0f);
        pc.m_TraceRadianceAtlasUAV = 0;
        pc.m_GlobalDFSRV           = 0;
        pc.m_CardResolution        = m_Resources->m_AtlasElementSize;
        pc.m_CardAtlasResolution   = m_Resolution;
        pc.m_CardDepthAtlasSRV     = 0;
        pc.m_CardNormalAtlasSRV    = 0;
        pc.m_AllCardObjDataId      = m_Resources->m_ObserveDeviceData->GetDescId();
        pc.m_AllMeshDFDataId       = meshDFList;
        pc.m_NumTotalCards         = m_Resources->m_MeshCardIndex.load();

        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.RadiosityGenPass",
            Internal::kIntShaderTableAyanami.RadiosityTraceCS, Vector3i{ 0, 1, 1 }, pc,
            [globalDFSRV, this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_GlobalDFSRV           = ctx.m_FgDesc->GetSRV(*globalDFSRV);
                data.m_CardDepthAtlasSRV     = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheTemporaryDepth);
                data.m_CardNormalAtlasSRV    = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);
                data.m_TraceRadianceAtlasUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneCacheIndirectRadianceAtlas);
                SetRootSignature(data, ctx);
            });

        pass.AddWriteResource(*m_Resources->m_RDGSceneCacheIndirectRadianceAtlas)
            .AddReadResource(*m_Resources->m_RDGSceneCacheNormalAtlas)
            .AddReadResource(*m_Resources->m_RDGSceneCacheTemporaryDepth)
            .AddReadResource(*globalDFSRV);

        return pass;
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

        auto& pass = AddComputePass<PushConst>(builder, "Ayanami.SurfaceCacheDirectLighting",
            Internal::kIntShaderTableAyanami.SurfaceCacheDirectLightCS,
            Vector3i{ (i32)tileGroups, (i32)tileGroups, (i32)cardGroups }, pc,
            [this](PushConst data, const FrameGraphPassContext& ctx) {
                data.m_DirectLightUAV = ctx.m_FgDesc->GetUAV(*m_Resources->m_RDGSceneDirectLighting);
                data.m_ShadowMaskSRV  = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneShadowVisibilityAtlas);
                data.m_NormalAtlasSRV = ctx.m_FgDesc->GetSRV(*m_Resources->m_RDGSceneCacheNormalAtlas);
                SetRootSignature(data, ctx);
            })
                         .AddWriteResource(*m_Resources->m_RDGSceneDirectLighting)
                         .AddReadResource(*m_Resources->m_RDGSceneCacheNormalAtlas)
                         .AddReadResource(*m_Resources->m_RDGSceneShadowVisibilityAtlas);
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
        return *m_Resources->m_RDGSceneDirectLighting;
    }

    IFRIT_APIDECL Graphics::Rhi::RhiBufferRef AyanamiTrivialSurfaceCacheManager::GetCardDataBuffer()
    {
        return m_Resources->m_ObserveDeviceData;
    }
    IFRIT_APIDECL u32 AyanamiTrivialSurfaceCacheManager::GetCardResolution() { return m_Resources->m_AtlasElementSize; }
    IFRIT_APIDECL u32 AyanamiTrivialSurfaceCacheManager::GetCardAtlasResolution() { return m_Resolution; }
    IFRIT_APIDECL u32 AyanamiTrivialSurfaceCacheManager::GetWorldMatsId()
    {
        return m_Resources->m_ObserveDeviceDataCoherentBindId->GetActiveId();
    }
    IFRIT_APIDECL u32 AyanamiTrivialSurfaceCacheManager::GetNumCards() { return m_Resources->m_MeshCardIndex.load(); }
} // namespace Ifrit::Runtime::Ayanami