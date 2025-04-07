
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

#include "ifrit/runtime/renderer/ayanami/AyanamiDebugger.h"
using namespace Ifrit::Graphics::Rhi;

namespace Ifrit::Runtime::Ayanami
{
    struct AyanamiDebuggerPrivate
    {
        RhiTextureRef m_RenderAtomicDepthAtlas = nullptr;
    };

    IFRIT_APIDECL AyanamiDebugger::AyanamiDebugger(Graphics::Rhi::RhiBackend* rhi) : m_Rhi(rhi)
    {
        m_Private = new AyanamiDebuggerPrivate();
    }
    IFRIT_APIDECL                  AyanamiDebugger::~AyanamiDebugger() { delete m_Private; }

    IFRIT_APIDECL ComputePassNode& AyanamiDebugger::RenderSceneFromCacheSurface(FrameGraphBuilder& builder,
        GPUTexture outTexture, u32 cardAlbedoAtlas, u32 cardNormalAtlas, u32 cardRadianceAtlas, u32 totalCards,
        u32 cardResolution, u32 cardAtlasResolution, u32 cardDataBuffer)
    {
    }
} // namespace Ifrit::Runtime::Ayanami