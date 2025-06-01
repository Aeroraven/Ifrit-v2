
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
#include "ifrit/runtime/common/Pch.h"

#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiDeferredShadingPrivate;

    class IFRIT_RUNTIME_API AyanamiDeferredShading
    {
    private:
        AyanamiDeferredShadingPrivate* m_Private = nullptr;
        Graphics::Rhi::RhiBackend*     m_Rhi     = nullptr;

    public:
        AyanamiDeferredShading(Graphics::Rhi::RhiBackend* rhi);
        virtual ~AyanamiDeferredShading();

        void InitContext(FrameGraphBuilder& builder, u32 rtWidth, u32 rtHeight);

        void RenderDeferredShadow(FrameGraphBuilder& builder, u32 perFrameCBV, FGBufferNodeRef shadowData,
            FGTextureNodeRef gbufferDepth, FGTextureNodeRef gbufferNormal, u32 totalLights);
        void RenderDeferredLighting(FrameGraphBuilder& builder, u32 perFrameCBV, FGTextureNodeRef gbufferDepth,
            FGTextureNodeRef gbufferNormal, FGTextureNodeRef gbufferAlbedo, FGBufferNodeRef shadowData,
            u32 totalLights);
        void ExperimentalFuse(FrameGraphBuilder& builder, FGTextureNodeRef gbufferAlbedo);

        FGTextureNodeRef GetRDGDirectShadowTexture() const;
        FGTextureNodeRef GetRDGDirectLightingTexture() const;
        FGTextureNodeRef GetRDGIndirectLightingTexture() const;
        FGTextureNodeRef GetRDGFinalLightingTexture() const;
        FGTextureNodeRef GetRDGLastFrameFinalLightingTexture() const;
    };
} // namespace Ifrit::Runtime::Ayanami