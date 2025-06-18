
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
#include "ifrit/runtime/material/ShaderRegistry.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime::Internal
{

#define DECLARE_VS(name) name "/VS"
#define DECLARE_FS(name) name "/FS"
#define DECLARE_CS(name) name "/CS"
#define DECLARE_MS(name) name "/MS"

#define SDEF IF_CONSTEXPR static const char*

    static struct InternalShaderTableSiro
    {
        SDEF TrivialPBDClothInit = DECLARE_CS("Siro/TrivialPBDCloth/Init");
    } kIntShaderTableSiro;

#undef SDEF

#undef DECLARE_VS
#undef DECLARE_FS
#undef DECLARE_CS
#undef DECLARE_MS

} // namespace Ifrit::Runtime::Internal