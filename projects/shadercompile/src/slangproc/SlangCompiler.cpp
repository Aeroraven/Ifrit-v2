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

#include <fstream>
namespace Ifrit::ShaderCompile::SlangProc
{

    void diagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
    {
        if (diagnosticsBlob != nullptr)
        {
            String diagnosticsString((const char*)diagnosticsBlob->getBufferPointer());
            iError("Slang diagnose: {}", diagnosticsString);
        }
    }

    SlangCompiler::SlangCompiler() : ShaderCompilerBase() {}

    SlangCompiler::~SlangCompiler() {}

    ShaderCompileOutput SlangCompiler::Compile(const ShaderCompileJob& job)
    {
        using Slang::ComPtr;
        ComPtr<slang::IGlobalSession> slangGlobalSession;

        iAssertion(
            slang::createGlobalSession(slangGlobalSession.writeRef()) >= 0, "Failed to create Slang global session");

        slang::SessionDesc sessionDesc = {};
        slang::TargetDesc  targetDesc  = {};
        targetDesc.format              = SLANG_SPIRV;
        targetDesc.profile             = slangGlobalSession->findProfile("spirv_1_5");
        targetDesc.flags               = 0;

        sessionDesc.targets                          = &targetDesc;
        sessionDesc.targetCount                      = 1;
        Array<slang::CompilerOptionEntry, 1> options = { { slang::CompilerOptionName::EmitSpirvDirectly,
            { slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr } } };
        sessionDesc.compilerOptionEntries            = options.data();
        sessionDesc.compilerOptionEntryCount         = 0; // options.size();

        ComPtr<slang::ISession> session;
        iAssertion(
            slangGlobalSession->createSession(sessionDesc, session.writeRef()) >= 0, "Failed to create Slang session");

        slang::IModule* slangModule = nullptr;
        {
            ComPtr<slang::IBlob> diagnosticBlob;
            slangModule = session->loadModuleFromSourceString(
                job.m_Name.c_str(), job.m_Name.c_str(), job.m_Source.m_Code.c_str(), diagnosticBlob.writeRef());
            diagnoseIfNeeded(diagnosticBlob);
            iAssertion(slangModule != nullptr, "Failed to load Slang module: {}", job.m_Name);
            // std::abort();
        }

        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            slangModule->findEntryPointByName(job.m_EntryPoint.c_str(), entryPoint.writeRef());
            if (!entryPoint)
            {
                iError("Failed to find entry point: {}", job.m_EntryPoint);
                std::abort();
            }
        }

        std::array<slang::IComponentType*, 2> componentTypes = { slangModule, entryPoint };
        Slang::ComPtr<slang::IComponentType>  composedProgram;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult                 result = session->createCompositeComponentType(
                componentTypes.data(), componentTypes.size(), composedProgram.writeRef(), diagnosticsBlob.writeRef());
            diagnoseIfNeeded(diagnosticsBlob);
            iAssertion(result >= 0, "Failed to create composite component type for slang module: {}", job.m_Name);
        }

        Slang::ComPtr<slang::IComponentType> linkedProgram;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
            diagnoseIfNeeded(diagnosticsBlob);
            iAssertion(result >= 0, "Failed to link program for slang module: {}", job.m_Name);
        }

        Slang::ComPtr<slang::IBlob> spirvCode;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult                 result =
                linkedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnosticsBlob.writeRef());
            diagnoseIfNeeded(diagnosticsBlob);
            iAssertion(result >= 0, "Failed to get SPIR-V code for slang module: {}, entry:{}, code:{}", job.m_Name,
                job.m_EntryPoint, (i32)result);
        }

        ShaderCompileOutput output;
        output.m_IR.m_Format = ShaderIRFormat::SpirV;
        output.m_IR.m_Data.CopyFromRaw(spirvCode->getBufferPointer(), spirvCode->getBufferSize());
        // output.m_Signature = # TODO

        // debug, write spirv to $cacheDir/job.m_Name.spv
        String        spirvFilePath = m_CachePath + "/Test.spv";
        std::ofstream spirvFile(spirvFilePath, std::ios::binary);
        if (spirvFile.is_open())
        {
            spirvFile.write((const char*)spirvCode->getBufferPointer(), spirvCode->getBufferSize());
            spirvFile.close();
            iDebug("Slang SPIR-V code written to: {}", spirvFilePath);
        }
        else
        {
            iError("Failed to write SPIR-V code to file: {}", spirvFilePath);
        }

        return output;
    }
} // namespace Ifrit::ShaderCompile::SlangProc