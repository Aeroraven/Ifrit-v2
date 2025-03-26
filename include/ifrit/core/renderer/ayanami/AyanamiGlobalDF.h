
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
        using GPUTexture = Graphics::Rhi::RhiTexture;
        Vector3f        m_worldBound;
        u32             m_clipmapSize;

        // I don't think this is a good design, but it's the most stupid and straightforward way to do it
        // That means ignoring paging, streaming and atlas.
        Ref<GPUTexture> m_clipmapTexture;
    };

    class IFRIT_APIDECL AyanamiGlobalDF : public Common::Utility::NonCopyable
    {
    private:
        Vec<Vec<AyanamiGlobalDFClipmap>> m_clipmaps;

    public:
        // AyanamiGlobalDF(const AyanamiRenderConfig &config);
        // ~AyanamiGlobalDF() = default;
    };

} // namespace Ifrit::Core::Ayanami
