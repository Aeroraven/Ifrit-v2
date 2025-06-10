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
#include "ifrit/shadercompile/base/ShaderCompileBase.h"

namespace Ifrit::ShaderCompile
{
    class IFRIT_SHADERCOMPILE_API ShaderCompileHelper
    {
    private:
        String                    m_IncludeBase;
        String                    m_CacheDir;
        ShaderCompileOptimization m_Optimization;

    public:
        ~ShaderCompileHelper();
        inline void         SetIncludeBase(const String& includeBase) { m_IncludeBase = includeBase; }
        inline void         SetCacheDir(const String& cacheDir) { m_CacheDir = cacheDir; }
        inline void         SetOptimization(ShaderCompileOptimization optimization) { m_Optimization = optimization; }
        ShaderCompileOutput CompileShaderFromSource(const ShaderCompileJob& job, ShaderIRFormat targetFormat);
        ShaderCompileOutput CompileShaderFromFile(const String& fileName, const String& entryPoint,
            const HashMap<String, String>& definitions, ShaderIRFormat targetFormat);
        ShaderSourceFormat  GetShaderSourceFormatFromFileName(const String& fileName);
    };
} // namespace Ifrit::ShaderCompile