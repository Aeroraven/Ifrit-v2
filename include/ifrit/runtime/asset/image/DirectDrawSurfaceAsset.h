
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

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
#include "ifrit/runtime/asset/TextureAsset.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/rhi/common/RhiForwardingTypes.h"

namespace Ifrit::Runtime
{
    class IFRIT_APIDECL DirectDrawSurfaceAsset : public TextureAsset
    {
    private:
        bool               m_loaded = false;
        IApplication*      m_app;
        RHI::RhiTextureRef m_texture = nullptr;

    public:
        using TextureAsset::TextureAsset;

        virtual RHI::RhiTextureRef GetTexture() override;
    };

} // namespace Ifrit::Runtime