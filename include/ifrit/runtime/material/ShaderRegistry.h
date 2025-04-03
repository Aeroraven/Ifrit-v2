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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "ifrit/core/PrecompiledHeaders.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime
{
    class IApplication;

    struct ShaderRegistryData;
    class IFRIT_APIDECL ShaderRegistry
    {
        using ShaderTp   = Graphics::Rhi::RhiShader;
        using ShaderType = Graphics::Rhi::RhiShaderStage;
        ShaderRegistryData* m_Data;

    public:
        ShaderRegistry(IApplication* app);
        ~ShaderRegistry();
        void      RegisterShader(const String& name, const String& path, const String& entry, ShaderType stage);
        void      WaitForShaderCompilations();

        // Note GetShader is a blocking call
        ShaderTp* GetShader(const String& name, u64 permutations);
    };

} // namespace Ifrit::Runtime