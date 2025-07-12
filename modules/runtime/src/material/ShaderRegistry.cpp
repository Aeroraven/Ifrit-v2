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
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/file/FileOps.h"

namespace Ifrit::Runtime
{

    struct ShaderRegistryData
    {
        using ShaderTp           = RHI::RhiShader;
        using ShaderCollectionTp = RHI::RhiShaderCollection;

        enum class ShaderStatus : u32
        {
            Uncompiled,
            Compiling,
            Compiled,
        };

        struct ShaderMapEntry
        {
            Ref<ShaderCollectionTp> m_Shader     = nullptr;
            TaskHandle              m_TaskHandle = nullptr;
            Atomic<ShaderStatus>    m_Status     = ShaderStatus::Uncompiled;
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

        auto   rhiCapability = m_Data->m_App->GetRhi()->GetCapabilities();
        if (!rhiCapability.m_MeshShaderEnabled && (stage == ShaderType::Mesh || stage == ShaderType::Task))
        {
            iWarn("ShaderRegistry: Shader `{}` requires mesh shader support. But your device does not support it,"
                  "or it is not enabled. Error will be raised when trying to use this shader.",
                sName);
            return;
        }
        if (!rhiCapability.m_HardwareRayTracingEnabled
            && (stage == ShaderType::RTRayGen || stage == ShaderType::RTMiss || stage == ShaderType::RTAnyHit
                || stage == ShaderType::RTClosestHit || stage == ShaderType::RTIntersection
                || stage == ShaderType::RTCallable))
        {
            iWarn(
                "ShaderRegistry: Shader `{}` requires hardware ray tracing support. But your device does not support it,"
                "or it is not enabled. Error will be raised when trying to use this shader.",
                sName);
            return;
        }

        if (!m_Data->m_ShaderMap.contains(sName))
        {
            m_Data->m_CompilingShaders.fetch_add(1, std::memory_order::acq_rel);
            m_Data->m_ShaderMap[sName].m_Status = ShaderRegistryData::ShaderStatus::Uncompiled;

            auto taskExecutor                       = GetFTaskScheduler();
            m_Data->m_ShaderMap[sName].m_TaskHandle = taskExecutor->EnqueueTask(
                [sPath, sName, sEntry, stage, this](Task* task, void* data) {
                    auto   fileExtension = sPath.substr(sPath.find_last_of('.') + 1);
                    String shaderPath;
                    auto   shaderType = RHI::RhiShaderSourceType::GLSLCode;
                    if (fileExtension == "glsl")
                    {
                        shaderPath = String(IFRIT_RUNTIME_SHARED_SHADER_PATH) + "/" + sPath;
                    }
                    else if (fileExtension == "slang")
                    {
                        shaderPath = String(IFRIT_RUNTIME_SHARED_SHADER_NEXT_PATH) + "/" + sPath;
                        shaderType = RHI::RhiShaderSourceType::SlangCode;
                    }
                    else
                    {
                        iErrorWithAbort("ShaderRegistry: Unsupported shader file extension: {}", fileExtension);
                    }

                    auto shaderCode = ReadTextFile(shaderPath);
                    if (shaderCode.size() == 0)
                    {
                        iErrorWithAbort("ShaderRegistry: Cannot read shader file {}", shaderPath.c_str());
                    }
                    auto shaderCodeVec = Vec<char>(shaderCode.begin(), shaderCode.end());
                    auto rhi           = m_Data->m_App->GetRhi();
                    auto shader        = rhi->CreateShader(sName, shaderCodeVec, sEntry, stage, shaderType);

                    m_Data->m_ShaderMap[sName].m_Shader = shader;
                    m_Data->m_ShaderMap[sName].m_Status.store(
                        ShaderRegistryData::ShaderStatus::Compiled, std::memory_order::release);
                    m_Data->m_CompilingShaders.fetch_sub(1, std::memory_order::acq_rel);
                    iInfo("ShaderRegistry: Compiled shader `{}`", sName);
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

    ShaderRegistry::ShaderTp* ShaderRegistry::GetShader(const ShaderVariantDesc& desc)
    {
        auto name         = desc.m_Name;
        auto permutations = desc.m_Defines;
        bool loadingTip   = false;

        if (m_Data->m_ShaderMap.contains(name))
        {
            auto& entry = m_Data->m_ShaderMap[name];
            while (entry.m_Status.load(std::memory_order::acquire) != ShaderRegistryData::ShaderStatus::Compiled)
            {
                if (!loadingTip)
                {
                    iInfo("ShaderRegistry: Waiting for shader `{}` to be compiled...", name);
                    loadingTip = true;
                }
                std::this_thread::yield();
            }
            return m_Data->m_ShaderMap[name].m_Shader->GetVariant(permutations);
        }
        else
        {
            iErrorWithAbort("ShaderRegistry: Shader {} not found", name);
            return nullptr;
        }
    }
} // namespace Ifrit::Runtime