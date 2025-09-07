#include "ifrit/shadercompile/slangproc/SlangCompiler.h"
#include "ifrit/core/logging/Logging.h"
#include "slang/include/slang-com-ptr.h"
#include "slang/include/slang.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/hal/HalHostConcurrency.h"
#include "ifrit/core/typing/EnumReflection.h"
#include "ifrit/shadercompile/helper/ShaderReflectionHelper.h"

#include "sha1/sha1.hpp"
#include <filesystem>
#include <fstream>
namespace Ifrit::ShaderCompile::SlangProc
{
    // ===== Shader Reflection =====

    void RecursiveDumpSlangType(
        slang::TypeLayoutReflection* typeLayout, ShaderReflectionHelper* reflHelper, const String& name, u32 baseOffset)
    {

        auto scalarTypeCvt = [](slang::TypeReflection::ScalarType scalar) -> EShaderScalarType {
            switch (scalar)
            {
                case slang::TypeReflection::ScalarType::Float32:
                    return EShaderScalarType::Float;
                case slang::TypeReflection::ScalarType::Int32:
                    return EShaderScalarType::Int32;
                case slang::TypeReflection::ScalarType::UInt32:
                    return EShaderScalarType::Uint32;
                case slang::TypeReflection::ScalarType::Float64:
                    return EShaderScalarType::Double;
                default:
                    IF_LOG_CRITICAL("SlangCompiler", "Unsupported scalar type");
                    return EShaderScalarType::Unknown;
            }
        };

        auto kind      = typeLayout->getKind();
        auto size      = typeLayout->getSize();
        auto tpname    = typeLayout->getType()->getName();
        auto elements  = typeLayout->getElementCount();
        auto underType = typeLayout->getElementTypeLayout();

        if (kind == slang::TypeReflection::Kind::Vector)
        {
            auto              scalarType = typeLayout->getElementTypeLayout()->getScalarType();
            EShaderScalarType destEnum   = scalarTypeCvt(scalarType);
            reflHelper->RegisterArithmeticShaderParams(name, destEnum, elements, false, baseOffset);
            return;
        }
        else if (kind == slang::TypeReflection::Kind::Scalar)
        {
            auto              scalarType = typeLayout->getScalarType();
            EShaderScalarType destEnum   = scalarTypeCvt(scalarType);
            reflHelper->RegisterArithmeticShaderParams(name, destEnum, 1, false, baseOffset);
            return;
        }
        else if (kind == slang::TypeReflection::Kind::Struct)
        {
            auto result = reflHelper->RegisterBindlessHandle(name, tpname, baseOffset);
            if (result == EShaderBindlessParamRegResult::Invalid)
            {
                auto members = typeLayout->getFieldCount();
                if (members == 1)
                {
                    auto memberLayout = typeLayout->getFieldByIndex(0);
                    RecursiveDumpSlangType(memberLayout->getTypeLayout(), reflHelper, name, baseOffset);
                }
                else
                {
                    IF_LOG_CRITICAL(
                        "SlangCompiler", "Struct type must be registered as bindless handle or have only one member");
                }
            }
        }
    }

    IF_NODISCARD ShaderReflectionData DumpSlangProgramReflectionData(slang::IComponentType* program)
    {
        ShaderReflectionHelper reflHelper;

        auto                   programLayout = program->getLayout();

        // Global Parameters
        auto                   paramCount2 = programLayout->getParameterCount();
        IF_LOG_DEBUG("SlangCompiler", "Global parameter count: {}", paramCount2);
        for (int i = 0; i < paramCount2; ++i)
        {

            auto paramLayout = programLayout->getParameterByIndex(i);

            auto param      = paramLayout->getVariable();
            auto paramName  = param->getName();
            auto paramType  = param->getType();
            auto typeLayout = paramLayout->getTypeLayout();
            auto category   = typeLayout->getParameterCategory();
            auto binding    = paramLayout->getBindingIndex();
            auto space      = paramLayout->getBindingSpace();

            auto isPushConstant = reflHelper.IsGlobalParameterPushConstant(paramName);
            if (isPushConstant)
            {
                auto members  = typeLayout->getFieldCount();
                auto typeName = typeLayout->getType()->getName();
                auto typeSize = typeLayout->getSize();
                reflHelper.SetRootConstantSize(typeSize);
                if (members > 0)
                {
                    for (int j = 0; j < members; ++j)
                    {
                        auto memberLayout  = typeLayout->getFieldByIndex(j);
                        auto member        = memberLayout->getVariable();
                        auto memberName    = member->getName();
                        auto memberType    = member->getType();
                        auto memberBinding = memberLayout->getBindingIndex();
                        auto memberSpace   = memberLayout->getBindingSpace();

                        RecursiveDumpSlangType(memberLayout->getTypeLayout(), &reflHelper, memberName, memberBinding);
                    }
                }
            }
        }
        return reflHelper.GetReflectionData();
    }

    // ===== Shader Compiler =====
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

    static HashMap<u32, FSlangCompilerPersistentData> sPersistentData;

    void                                              DiagnoseIfNeeded(slang::IBlob* diagnosticsBlob)
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

        auto sourceCode = job.mSource.mCode;
        sourceCode      = "#define IFSHADER_VULKAN 1\n" + sourceCode;
        for (const auto& [key, value] : job.mDefinitions)
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
                { slang::CompilerOptionValueKind::String, 0, 0, mIncludeBase.c_str(), nullptr } },
            { slang::CompilerOptionName::Optimization, { slang::CompilerOptionValueKind::Int, 3, 0, nullptr, nullptr } }

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
                job.mName.c_str(), job.mName.c_str(), sourceCode.c_str(), diagnosticBlob.writeRef());
            DiagnoseIfNeeded(diagnosticBlob);
            IF_LOG_ASSERTION("SlangCompiler", slangModule != nullptr, "Failed to load Slang module: {}", job.mName);
            // std::abort();
        }

        ComPtr<slang::IBlob> serializedModule;
        {
            SlangResult result = slangModule->serialize(serializedModule.writeRef());
            IF_LOG_ASSERTION(
                "SlangCompiler", result >= 0, "Failed to serialize Slang module: {}, code:{}", job.mName, (i32)result);
        }
        String serializedModuleStr;
        serializedModuleStr.resize(serializedModule->getBufferSize());
        memcpy(serializedModuleStr.data(), serializedModule->getBufferPointer(), serializedModule->getBufferSize());

        SHA1 sha1;
        sha1.update(serializedModuleStr);
        String moduleHash = sha1.final();
        // iDebug("Slang module {} hash: {}", job.m_Name, moduleHash);

        String cachedModulePath = mCachePath + "/ifritsc.slang.shader." + moduleHash + ".cache";
        if (std::filesystem::exists(cachedModulePath) && false)
        {
            // IF_LOG_DEBUG("Slang", "Using cached Slang module: {} for {}", cachedModulePath, job.m_Name);
            ShaderCompileOutput output;
            output.mIR.mFormat = EShaderIRFormat::SpirV;
            std::ifstream file(cachedModulePath, std::ios::binary);
            if (file)
            {
                file.seekg(0, std::ios::end);
                size_t size = file.tellg();
                file.seekg(0, std::ios::beg);
                Vec<u8> data;
                data.resize(size);
                file.read(reinterpret_cast<char*>(data.data()), size);

                output.mIR.mData.CopyFromRaw(data.data(), SizeCast<u32>(data.size()));
                output.mIR.mFormat = EShaderIRFormat::SpirV;
            }
            else
            {
                IF_LOG_CRITICAL("SlangCompiler", "Failed to read cached Slang module: {}", cachedModulePath);
                std::abort();
            }
            output.mSignature = moduleHash;
            return output;
        }

        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            slangModule->findEntryPointByName(job.mEntryPoint.c_str(), entryPoint.writeRef());
            if (!entryPoint)
            {
                IF_LOG_CRITICAL("SlangCompiler", "Failed to find entry point: {}", job.mEntryPoint);
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
                "Failed to create composite component type for slang module: {}", job.mName);
        }

        Slang::ComPtr<slang::IComponentType> linkedProgram;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
            DiagnoseIfNeeded(diagnosticsBlob);
            IF_LOG_ASSERTION("SlangCompiler", result >= 0, "Failed to link program for slang module: {}", job.mName);
        }

        auto                        reflData = DumpSlangProgramReflectionData(linkedProgram);

        Slang::ComPtr<slang::IBlob> spirvCode;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            SlangResult                 result =
                linkedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnosticsBlob.writeRef());
            DiagnoseIfNeeded(diagnosticsBlob);
            IF_LOG_ASSERTION("SlangCompiler", result >= 0,
                "Failed to get SPIR-V code for slang module: {}, entry:{}, code:{}", job.mName, job.mEntryPoint,
                (i32)result);
        }

        ShaderCompileOutput output;
        output.mIR.mFormat = EShaderIRFormat::SpirV;
        output.mIR.mData.CopyFromRaw(spirvCode->getBufferPointer(), SizeCast<u32>(spirvCode->getBufferSize()));
        output.mSignature = moduleHash;
        output.mReflData  = reflData;

        // write to cache
        if (mCachePath.size())
        {
            std::ofstream cacheFile(cachedModulePath, std::ios::binary);
            if (cacheFile)
            {
                cacheFile.write(reinterpret_cast<const char*>(output.mIR.mData.GetData()), output.mIR.mData.GetSize());
                cacheFile.close();
            }
            else
            {
                IF_LOG_CRITICAL("SlangCompiler", "Failed to write cached Slang module: {}", cachedModulePath);
            }
        }


        return output;
    }
} // namespace Ifrit::ShaderCompile::SlangProc