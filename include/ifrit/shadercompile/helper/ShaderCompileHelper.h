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
#include "ifrit/core/base/containers/Maps.h"

namespace Ifrit::ShaderCompile
{
    class IFRIT_SHADERCOMPILE_API ShaderCompileHelper
    {
    private:
        String                     mIncludeBase;
        String                     mCacheDir;
        EShaderCompileOptimization mOptimization;

    public:
        ~ShaderCompileHelper();
        inline void         SetIncludeBase(const String& includeBase) { mIncludeBase = includeBase; }
        inline void         SetCacheDir(const String& cacheDir) { mCacheDir = cacheDir; }
        inline void         SetOptimization(EShaderCompileOptimization optimization) { mOptimization = optimization; }
        ShaderCompileOutput CompileShaderFromSource(const ShaderCompileJob& job, EShaderIRFormat targetFormat);
        ShaderCompileOutput CompileShaderFromFile(const String& fileName, const String& entryPoint,
            const THashMap<String, String>& definitions, EShaderIRFormat targetFormat);
        EShaderSourceFormat GetEShaderSourceFormatFromFileName(const String& fileName);
    };
} // namespace Ifrit::ShaderCompile