
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

#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/asset/ShaderAsset.h"
#include <fstream>
namespace Ifrit::Runtime
{

    // Shader class
    IFRIT_APIDECL ShaderAsset::ShaderRef* ShaderAsset::LoadShader(const Vec<String>& permutations)
    {
        if (m_loaded)
        {
            return m_selfData->GetVariant(permutations);
        }
        else
        {
            m_loaded = true;
            std::ifstream       file(mMetadata.mExternalPath, std::ios::binary);
            Vec<char>           data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

            auto                rhi = GetActiveApplication()->GetRhi();
            RHI::RhiShaderStage stage;
            auto                fileName = mMetadata.mExternalPath;
            // endswith .vert.glsl
            auto                endsWith = [](const String& str, const String& suffix) {
                return str.size() >= suffix.size()
                    && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            if (endsWith(fileName, ".vert.glsl"))
            {
                stage = RHI::RhiShaderStage::Vertex;
            }
            else if (endsWith(fileName, ".frag.glsl"))
            {
                stage = RHI::RhiShaderStage::Fragment;
            }
            else if (endsWith(fileName, ".comp.glsl"))
            {
                stage = RHI::RhiShaderStage::Compute;
            }
            else if (endsWith(fileName, ".mesh.glsl"))
            {
                stage = RHI::RhiShaderStage::Mesh;
            }
            else if (endsWith(fileName, ".task.glsl"))
            {
                stage = RHI::RhiShaderStage::Task;
            }
            else
            {
                throw std::runtime_error("Unknown shader stage");
            }

            auto p = rhi->CreateShader(fileName, data, "main", stage, RHI::RhiShaderSourceType::GLSLCode);
            // TODO: eliminate raw pointer
            m_selfData = p;
            return m_selfData->GetVariant(permutations);
        }
    }

} // namespace Ifrit::Runtime