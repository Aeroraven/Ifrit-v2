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

#include "ifrit/shadercompile/base/ShaderCompileBase.h"

namespace Ifrit::ShaderCompile
{

    IFRIT_APIDECL void ShaderCompilerBase::SetCachePath(const String& cacheDir) { m_CachePath = cacheDir; }

    IFRIT_APIDECL void ShaderCompilerBase::SetIncludeBase(const String& includeBase) { m_IncludeBase = includeBase; }

    IFRIT_APIDECL void ShaderCompilerBase::SetOptimization(ShaderCompileOptimization optimization)
    {
        m_Optimization = optimization;
    }

} // namespace Ifrit::ShaderCompile