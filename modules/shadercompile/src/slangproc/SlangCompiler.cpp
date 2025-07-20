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

#include "ifrit/shadercompile/slangproc/SlangCompiler.h"
#include "ifrit/core/logging/Logging.h"
#include "slang/include/slang-com-ptr.h"
#include "slang/include/slang.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/hal/HalHostConcurrency.h"

#include "sha1/sha1.hpp"
#include <filesystem>
#include <fstream>
namespace Ifrit::ShaderCompile::SlangProc
{
    struct FSlangCompilerPersistentData
    {
    private:
        Slang::ComPtr<slang::IGlobalSession> m_SlangGlobalSession = nullptr;

    public:
        FSlangCompilerPersistentData() {}
        Slang::ComPtr<slang::IGlobalSession> GetGlobalSession()
        {
            if (!m_SlangGlobalSession)
            {
                slang::createGlobalSession(m_SlangGlobalSession.writeRef());
                IF_LOG_ASSERTION(
                    "SlangCompiler", m_SlangGlobalSession != nullptr, "Failed to create Slang global session");
            }
            return m_SlangGlobalSession;
        }
    };

    static Vec<FSlangCompilerPersistentData> sPersistentData(HAL::GetMaxThreadLimit());

    void                                     DiagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
    {
        if (diagnosticsBlob != nullptr)
        {
            String diagnosticsString((const char*)diagnosticsBlob->getBufferPointer());
            IF_LOG_CRITICAL("SlangCompiler", "Slang diagnose: {}", diagnosticsString);
        }
    }

    SlangCompiler::SlangCompiler() : ShaderCompilerBase() {}

    SlangCompiler::~SlangCompiler() {}

    ShaderCompileOutput SlangCompiler::Compile(const ShaderCompileJob& job)
    {
        using Slang::ComPtr;

        auto slangGlobalSession = sPersistentData[HAL::GetCurrentThreadId()].GetGlobalSession();

        auto sourceCode = job.m_Source.m_Code;
        sourceCode      = "#define IFSHADER_VULKAN 1\n" + sourceCode;
        for (const auto& [key, value] : job.m_Definitions)
        {
            sourceCode = "#define " + key + " " + value + "\n" + sourceCode;
        }

        slang::SessionDesc sessionDesc = {};
        slang::TargetDesc  targetDesc  = {};
        targetDesc.format              = SLANG_SPIRV;
        targetDesc.profile             = slangGlobalSession->findProfile("spirv_1_5");
        targetDesc.flags               = 0;

        sessionDesc.targets                     = &targetDesc;
        sessionDesc.targetCount                 = 1;
        sessionDesc.defaultMatrixLayoutMode     = SlangMatrixLayoutMode::SLANG_MATRIX_LAYOUT_COLUMN_MAJOR;
        auto capNonUniformBallot                = slangGlobalSession->findCapability("spvGroupNonUniformBallot");
        Vec<slang::CompilerOptionEntry> options = {
            { slang::CompilerOptionName::EmitSpirvDirectly,
                { slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr } },
            { slang::CompilerOptionName::Capability,
                { slang::CompilerOptionValueKind::Int, capNonUniformBallot, 0, nullptr, nullptr } },
            { slang::CompilerOptionName::Include,
                { slang::CompilerOptionValueKind::String, 0, 0, m_IncludeBase.c_str(), nullptr } },

        };

        sessionDesc.compilerOptionEntries    = options.data();
        sessionDesc.compilerOptionEntryCount = SizeCast<u32>(options.size());

        ComPtr<slang::ISession> session;
        IF_LOG_ASSERTION("SlangCompiler", slangGlobalSession->createSession(sessionDesc, session.writeRef()) >= 0,
            "Failed to create Slang session");

        slang::IModule* slangModule = nullptr;
        {
            ComPtr<slang::IBlob> diagnosticBlob;
            slangModule = session->loadModuleFromSourceString(
                job.m_Name.c_str(), job.m_Name.c_str(), sourceCode.c_str(), diagnosticBlob.writeRef());
            DiagnoseIfNeeded(diagnosticBlob);
            IF_LOG_ASSERTION("SlangCompiler", slangModule != nullptr, "Failed to load Slang module: {}", job.m_Name);
            // std::abort();
        }

        ComPtr<slang::IBlob> serializedModule;
        {
            SlangResult result = slangModule->serialize(serializedModule.writeRef());
            IF_LOG_ASSERTION(
                "SlangCompiler", result >= 0, "Failed to serialize Slang module: {}, code:{}", job.m_Name, (i32)result);
        }
        String serializedModuleStr;
        serializedModuleStr.resize(serializedModule->getBufferSize());
        memcpy(serializedModuleStr.data(), serializedModule->getBufferPointer(), serializedModule->getBufferSize());

        SHA1 sha1;
        sha1.update(serializedModuleStr);
        String moduleHash = sha1.final();
        // iDebug("Slang module {} hash: {}", job.m_Name, moduleHash);

        String cachedModulePath = m_CachePath + "/ifritsc.slang.shader." + moduleHash + ".cache";
        if (std::filesystem::exists(cachedModulePath))
        {
            // iDebug("Using cached Slang module: {}", cachedModulePath);
            ShaderCompileOutput output;
            output.m_IR.m_Format = ShaderIRFormat::SpirV;
            std::ifstream file(cachedModulePath, std::ios::binary);
            if (file)
            {
                file.seekg(0, std::ios::end);
                size_t size = file.tellg();
                file.seekg(0, std::ios::beg);
                Vec<u8> data;
                data.resize(size);
                file.read(reinterpret_cast<char*>(data.data()), size);

                output.m_IR.m_Data.CopyFromRaw(data.data(), SizeCast<u32>(data.size()));
                output.m_IR.m_Format = ShaderIRFormat::SpirV;
            }
            else
            {
                IF_LOG_CRITICAL("SlangCompiler", "Failed to read cached Slang module: {}", cachedModulePath);
                std::abort();
            }
            output.m_Signature = moduleHash;
            return output;
        }

        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            slangModule->findEntryPointByName(job.m_EntryPoint.c_str(), entryPoint.writeRef());
            if (!entryPoint)
            {
                IF_LOG_CRITICAL("SlangCompiler", "Failed to find entry point: {}", job.m_EntryPoint);
                std::abort();
            }
        }

        std::array<slang::IComponentType*, 2> componentTypes = { slangModule, entryPoint };
        Slang::ComPtr<slang::IComponentType>  composedProgram;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult                 result = session->createCompositeComponentType(
                componentTypes.data(), componentTypes.size(), composedProgram.writeRef(), diagnosticsBlob.writeRef());
            DiagnoseIfNeeded(diagnosticsBlob);
            IF_LOG_ASSERTION("SlangCompiler", result >= 0,
                "Failed to create composite component type for slang module: {}", job.m_Name);
        }

        Slang::ComPtr<slang::IComponentType> linkedProgram;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
            DiagnoseIfNeeded(diagnosticsBlob);
            IF_LOG_ASSERTION("SlangCompiler", result >= 0, "Failed to link program for slang module: {}", job.m_Name);
        }

        Slang::ComPtr<slang::IBlob> spirvCode;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult                 result =
                linkedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnosticsBlob.writeRef());
            DiagnoseIfNeeded(diagnosticsBlob);
            IF_LOG_ASSERTION("SlangCompiler", result >= 0,
                "Failed to get SPIR-V code for slang module: {}, entry:{}, code:{}", job.m_Name, job.m_EntryPoint,
                (i32)result);
        }

        ShaderCompileOutput output;
        output.m_IR.m_Format = ShaderIRFormat::SpirV;
        output.m_IR.m_Data.CopyFromRaw(spirvCode->getBufferPointer(), SizeCast<u32>(spirvCode->getBufferSize()));
        output.m_Signature = moduleHash;

        // write to cache
        std::ofstream cacheFile(cachedModulePath, std::ios::binary);
        if (cacheFile)
        {
            cacheFile.write(reinterpret_cast<const char*>(output.m_IR.m_Data.GetData()), output.m_IR.m_Data.GetSize());
            cacheFile.close();
            // iDebug("Cached Slang module: {}", cachedModulePath);
        }
        else
        {
            IF_LOG_CRITICAL("SlangCompiler", "Failed to write cached Slang module: {}", cachedModulePath);
        }

        return output;
    }
} // namespace Ifrit::ShaderCompile::SlangProc