
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
#include "ifrit/core/math/VectorDefs.h"

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiDistanceFieldLightingPrivate;
    class IFRIT_APIDECL AyanamiDistanceFieldLighting
    {
    private:
        Graphics::Rhi::RhiBackend*           m_Rhi     = nullptr;
        AyanamiDistanceFieldLightingPrivate* m_Private = nullptr;

    public:
        AyanamiDistanceFieldLighting(Graphics::Rhi::RhiBackend* rhi);
        ~AyanamiDistanceFieldLighting();

        GraphicsPassNode& DistanceFieldShadowTileScatter(FrameGraphBuilder& builder, u32 meshDfList, u32 totalMeshDfs,
            Vector4f sceneBound, Vector3f lightDir, u32 tileSize);

        GraphicsPassNode& DistanceFieldShadowRender(FrameGraphBuilder& builder, u32 meshDfList, u32 totalMeshDfs,
            u32 depthSRV, u32 perframe, Graphics::Rhi::RhiRenderTargets* rts, Vector4f sceneBound, Vector3f lightDir,
            u32 tileSize, float softness);
    };
} // namespace Ifrit::Runtime::Ayanami
