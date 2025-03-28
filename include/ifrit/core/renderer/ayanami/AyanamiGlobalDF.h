
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
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/math/LinalgOps.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/renderer/ayanami/AyanamiRenderConfig.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core::Ayanami
{

    struct IFRIT_APIDECL AyanamiGlobalDFClipmap : public Common::Utility::NonCopyable
    {
        using GPUTexture    = Graphics::Rhi::RhiTextureRef;
        using GPUSamplerRef = Graphics::Rhi::RhiSamplerRef;
        using GPUResId      = Graphics::Rhi::RhiDescHandleLegacy;
        Vector3f      m_worldBoundMin;
        Vector3f      m_worldBoundMax;
        u32           m_clipmapSize;

        // I don't think this is a good design, but it's the most stupid and straightforward way to do it
        // That means ignoring paging, streaming and atlas.
        GPUTexture    m_clipmapTexture;
        GPUSamplerRef m_clipmapSampler;
        Ref<GPUResId> m_clipmapSRV;

        AyanamiGlobalDFClipmap() {}
    };

    class IFRIT_APIDECL AyanamiGlobalDF : public Common::Utility::NonCopyable
    {
    private:
        using GPUTexture = Graphics::Rhi::RhiTextureRef;

        Graphics::Rhi::RhiBackend*        m_rhi;
        Graphics::Rhi::RhiComputePass*    m_updateClipmapPass = nullptr;
        Graphics::Rhi::RhiComputePass*    m_raymarchPass      = nullptr;
        Vec<Uref<AyanamiGlobalDFClipmap>> m_TestClipMaps;

    public:
        AyanamiGlobalDF(const AyanamiRenderConfig& config, Graphics::Rhi::RhiBackend* rhi);
        ~AyanamiGlobalDF() = default;

        void       AddClipmapUpdate(const Graphics::Rhi::RhiCommandList* cmdList, u32 clipmapLevel, u32 perFrameDataId,
                  u32 numMeshes, u32 meshDFListId);

        void       AddRayMarchPass(const Graphics::Rhi::RhiCommandList* cmdList, u32 clipmapLevel, u32 perFrameDataId,
                  u32 outTextureId, Vector2u outTextureSize);
        GPUTexture GetClipmapVolume(u32 clipmapLevel);
    };

} // namespace Ifrit::Core::Ayanami
