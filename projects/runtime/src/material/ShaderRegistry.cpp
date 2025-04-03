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

#include "ifrit/runtime/material/ShaderRegistry.h"
#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/core/global/GlobalInstances.h"
#include "ifrit/runtime/base/ApplicationInterface.h"

namespace Ifrit::Runtime
{

    struct ShaderRegistryData
    {
        using ShaderTp = Graphics::Rhi::RhiShader;

        enum class ShaderStatus : u32
        {
            Uncompiled,
            Compiling,
            Compiled,
        };

        struct ShaderMapEntry
        {
            ShaderTp*            m_Shader     = nullptr;
            TaskHandle           m_TaskHandle = nullptr;
            Atomic<ShaderStatus> m_Status     = ShaderStatus::Uncompiled;
        };

        IApplication*                   m_App;
        Atomic<u32>                     m_CompilingShaders = 0;
        HashMap<String, ShaderMapEntry> m_ShaderMap;
    };

    IFRIT_APIDECL ShaderRegistry::ShaderRegistry(IApplication* app) : m_Data(new ShaderRegistryData)
    {
        m_Data->m_App = app;
    }
    IFRIT_APIDECL ShaderRegistry::~ShaderRegistry()
    {
        delete m_Data;
        m_Data = nullptr;
    }

    void ShaderRegistry::RegisterShader(const String& name, const String& path, const String& entry, ShaderType stage)
    {
        // Warning: This function is not thread-safe.
        String sName  = name;
        String sPath  = path;
        String sEntry = entry;

        if (!m_Data->m_ShaderMap.contains(sName))
        {
            m_Data->m_CompilingShaders.fetch_add(1, std::memory_order::acq_rel);
            m_Data->m_ShaderMap[sName].m_Status = ShaderRegistryData::ShaderStatus::Uncompiled;

            auto taskExecutor  = GetTaskScheduler();
            m_Data->m_ShaderMap[sName].m_TaskHandle = taskExecutor->EnqueueTask(
                [sPath,sName,sEntry,stage,this](Task* task, void* data) {
                    auto shaderPath    = String(IFRIT_RUNTIME_SHARED_SHADER_PATH) + "/" + sPath;
                    auto shaderCode    = ReadTextFile(shaderPath);
                    if (shaderCode.size() == 0)
                    {
                        printf("Cannot read file %s\n", shaderPath.c_str());
                    }
                    auto shaderCodeVec = Vec<char>(shaderCode.begin(), shaderCode.end());
                    auto rhi           = m_Data->m_App->GetRhi();
                    auto shader        = rhi->CreateShader(
                        sName, shaderCodeVec, sEntry, stage, Graphics::Rhi::RhiShaderSourceType::GLSLCode);

                    m_Data->m_ShaderMap[sName].m_Shader = shader;
                    m_Data->m_ShaderMap[sName].m_Status.store(
                        ShaderRegistryData::ShaderStatus::Compiled, std::memory_order::release);
                    m_Data->m_CompilingShaders.fetch_sub(1, std::memory_order::acq_rel);
                },
                {}, nullptr);
        }
    }
    void ShaderRegistry::WaitForShaderCompilations()
    {
        while (m_Data->m_CompilingShaders.load(std::memory_order::acquire) > 0)
        {
            std::this_thread::yield();
        }
    }

    ShaderRegistry::ShaderTp* ShaderRegistry::GetShader(const String& name, u64 permutations)
    {
        if (m_Data->m_ShaderMap.contains(name))
        {
            auto& entry = m_Data->m_ShaderMap[name];
            while (entry.m_Status.load(std::memory_order::acquire) != ShaderRegistryData::ShaderStatus::Compiled)
            {
                std::this_thread::yield();
            }
            return m_Data->m_ShaderMap[name].m_Shader;
        }
        else
        {
            iError("Shader {} not found", name);
            return nullptr;
        }
    }
} // namespace Ifrit::Runtime