
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

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/runtime/base/Scene.h"
#include "AyanamiRenderConfig.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiTrivialSurfaceCacheManagerResource;

    // Now, we not consider the coverage of surfels to generate mesh cards.
    // Here, we only use 6 faces to make a mesh card
    class IFRIT_APIDECL AyanamiTrivialSurfaceCacheManager
    {
    private:
        IApplication*                              m_App;
        u32                                        m_Resolution;
        AyanamiTrivialSurfaceCacheManagerResource* m_Resources = nullptr;

    public:
        AyanamiTrivialSurfaceCacheManager(const AyanamiRenderConfig& config, IApplication* app);
        ~AyanamiTrivialSurfaceCacheManager();

        void              UpdateSceneCache(Scene* scene);
        void              PrepareImmutableResource();
        void              InitContext(FrameGraphBuilder& builder);

        GraphicsPassNode& UpdateSurfaceCacheAtlas(FrameGraphBuilder& builder);
        ComputePassNode&  UpdateShadowVisibilityAtlas(FrameGraphBuilder& builder, Scene* scene);
        void              UpdateSurfaceModelMatrix();
        ComputePassNode&  UpdateIndirectRadianceCacheAtlas(
             FrameGraphBuilder& builder, Scene* scene, FGTextureNodeRef globalDFSRV, u32 meshDFList);
        void                        UpdateDirectLighting(FrameGraphBuilder& builder, u32 meshDFList, Vector3f lightDir);

        FGTextureNode&              GetRDGAlbedoAtlas();
        FGTextureNode&              GetRDGNormalAtlas();
        FGTextureNode&              GetRDGDepthAtlas();
        FGTextureNode&              GetRDGShadowVisibilityAtlas();
        FGTextureNode&              GetRDGTracedRadianceAtlas();
        FGTextureNode&              GetRDGDirectLightingAtlas();

        Graphics::Rhi::RhiBufferRef GetCardDataBuffer();
        u32                         GetCardResolution();
        u32                         GetCardAtlasResolution();
        u32                         GetWorldMatsId();
        u32                         GetNumCards();
    };
} // namespace Ifrit::Runtime::Ayanami