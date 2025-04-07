
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
#include "ifrit/core/math/LinalgOps.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiRenderConfig.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"

namespace Ifrit::Runtime::Ayanami
{

    struct IFRIT_APIDECL AyanamiGlobalDFClipmap : public NonCopyable
    {
        using GPUTexture    = Graphics::Rhi::RhiTextureRef;
        using GPUSamplerRef = Graphics::Rhi::RhiSamplerRef;
        using GPUResId      = Graphics::Rhi::RhiDescHandleLegacy;
        using GPUBuffer     = Graphics::Rhi::RhiBufferRef;

        Vector3f      m_worldBoundMin;
        Vector3f      m_worldBoundMax;
        u32           m_clipmapSize;

        // I don't think this is a good design, but it's the most stupid and straightforward way to do it
        // That means ignoring paging, streaming and atlas.
        GPUTexture    m_clipmapTexture;
        GPUSamplerRef m_clipmapSampler;
        Ref<GPUResId> m_clipmapSRV;

        GPUBuffer     m_objectGridBuffer;
        u32           m_VoxelsPerWidth;
        AyanamiGlobalDFClipmap() {}
    };

    class IFRIT_APIDECL AyanamiGlobalDF : public NonCopyable
    {
    private:
        using GPUTexture = Graphics::Rhi::RhiTextureRef;

        IApplication*                     m_app;
        Graphics::Rhi::RhiComputePass*    m_updateClipmapPass = nullptr;
        Graphics::Rhi::RhiComputePass*    m_raymarchPass      = nullptr;
        Vec<Uref<AyanamiGlobalDFClipmap>> m_TestClipMaps;

    public:
        AyanamiGlobalDF(const AyanamiRenderConfig& config, IApplication* m_app);
        ~AyanamiGlobalDF() = default;

        ComputePassNode& AddClipmapUpdate(
            FrameGraphBuilder& builder, u32 clipmapLevel, u32 perFrameDataId, u32 numMeshes, u32 meshDFListId);

        ComputePassNode& AddRayMarchPass(FrameGraphBuilder& builder, u32 clipmapLevel, u32 perFrameDataId,
            u32 outTextureId, Vector2u outTextureSize);

        ComputePassNode& AddObjectGridCompositionPass(
            FrameGraphBuilder& builder, u32 clipmapLevel, u32 numMeshes, u32 meshDFListId);

        GPUTexture GetClipmapVolume(u32 clipmapLevel);
    };

} // namespace Ifrit::Runtime::Ayanami
