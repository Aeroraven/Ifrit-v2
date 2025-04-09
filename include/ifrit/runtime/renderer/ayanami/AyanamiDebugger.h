
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
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiDebuggerPrivate;
    class IFRIT_RUNTIME_API AyanamiDebugger
    {
    private:
        using GPUBuffer  = Graphics::Rhi::RhiBufferRef;
        using GPUTexture = Graphics::Rhi::RhiTextureRef;

        AyanamiDebuggerPrivate*    m_Private = nullptr;
        Graphics::Rhi::RhiBackend* m_Rhi     = nullptr;

    public:
        AyanamiDebugger(Graphics::Rhi::RhiBackend* rhi);
        ~AyanamiDebugger();

        ComputePassNode& RenderSceneFromCacheSurface(FrameGraphBuilder& builder, FGTextureNodeRef outputTexture,
            FGTextureNodeRef cardAlbedoAtlas, FGTextureNodeRef cardNormalAtlas, FGTextureNodeRef cardRadianceAtlas,
            FGTextureNodeRef cardDepthAtlas, u32 totalCards, u32 cardResolution, u32 cardAtlasResolution,
            u32 cardDataBuffer, u32 perFrameId, u32 meshDfListId);
    };
} // namespace Ifrit::Runtime::Ayanami