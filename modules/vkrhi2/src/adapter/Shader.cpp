#include "ifrit/vkrhi2/adapter/Shader.h"
#include "ifrit/core/file/FileOps.h"
#include "ifrit/core/algo/StlStringUtils.h"
#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/core/console/ConsoleObject.h"
#include "ifrit/rhi/common/RhiShaderResource.h"

#include "ifrit/shadercompile/helper/ShaderCompileHelper.h"
#include "ifrit/vkrhi2/util/Log.h"
#include "ifrit/core/math/VectorDefs.h"

namespace Ifrit::RHI::VulkanRHI2
{

    // ===== Shader Variant =====

    struct VA_ShaderVariantInternal
    {
        VA_Device*                          mDevice       = nullptr;
        VkShaderModule                      mShaderModule = VK_NULL_HANDLE;
        VkPipelineShaderStageCreateInfo     mStageCI{};
        u64                                 mSignatureHash = 0;
        ShaderCompile::ShaderReflectionData mReflection;
        String                              mEntryPoint;
    };

    VA_ShaderVariant::VA_ShaderVariant(VA_Device* device, const VA_ShaderVariantCI& ci, void* reflData)
    {
        mData              = new VA_ShaderVariantInternal();
        mData->mDevice     = device;
        mData->mReflection = *(ShaderCompile::ShaderReflectionData*)reflData;

        VkShaderModuleCreateInfo moduleCI{};
        if (ci.mStage == RHI::ERhiShaderStage::Vertex)
            mData->mStageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
        else if (ci.mStage == RHI::ERhiShaderStage::Fragment)
            mData->mStageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        else if (ci.mStage == RHI::ERhiShaderStage::Compute)
            mData->mStageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        else if (ci.mStage == RHI::ERhiShaderStage::Mesh)
            mData->mStageCI.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        else if (ci.mStage == RHI::ERhiShaderStage::Task)
            mData->mStageCI.stage = VK_SHADER_STAGE_TASK_BIT_EXT;
        else if (ci.mStage == RHI::ERhiShaderStage::RTRayGen)
            mData->mStageCI.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        else if (ci.mStage == RHI::ERhiShaderStage::RTClosestHit)
            mData->mStageCI.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        else if (ci.mStage == RHI::ERhiShaderStage::RTMiss)
            mData->mStageCI.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        else if (ci.mStage == RHI::ERhiShaderStage::RTAnyHit)
            mData->mStageCI.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
        else if (ci.mStage == RHI::ERhiShaderStage::RTIntersection)
            mData->mStageCI.stage = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

        VkDevice deviceNative = device->GetVulkanDevice();
        moduleCI.sType        = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCI.pCode        = reinterpret_cast<const u32*>(ci.mIRCode.data());
        moduleCI.codeSize     = SizeCast<u32>(ci.mIRCode.size() * sizeof(char));

        VA_AssertResult(vkCreateShaderModule(deviceNative, &moduleCI, nullptr, &mData->mShaderModule),
            "Failed to create shader module");

        mData->mEntryPoint = ci.mEntryPoint;

        mData->mStageCI.module = mData->mShaderModule;
        mData->mStageCI.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        mData->mStageCI.pName  = mData->mEntryPoint.c_str();
        mData->mStageCI.pNext  = nullptr;
        mData->mStageCI.flags  = 0;
    }

    VA_ShaderVariant::~VA_ShaderVariant()
    {
        IF_LOG_DEBUG("VA_ShaderVariant", "Destroying shader variant: {}", mData->mEntryPoint);
        if (mData->mShaderModule != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(mData->mDevice->GetVulkanDevice(), mData->mShaderModule, nullptr);
            mData->mShaderModule = VK_NULL_HANDLE;
        }
        delete mData;
    }

    IFRIT_APIDECL u64          VA_ShaderVariant::GetSignatureHash() const { return (u64)(mData->mShaderModule); }
    IFRIT_APIDECL RhiRawHandle VA_ShaderVariant::GetRawHandle() const
    {
        return reinterpret_cast<RhiRawHandle>(mData->mShaderModule);
    }
    IFRIT_APIDECL bool VA_ShaderVariant::ValidateShaderParameters(const RhiShaderParameter& params)
    {
        auto checkValidAnyCast = [](const std::any& val, ShaderCompile::EShaderReflDescriptors type) {
            switch (type)
            {
                case ShaderCompile::EShaderReflDescriptors::DataBool:
                    return val.type() == typeid(bool);
                case ShaderCompile::EShaderReflDescriptors::DataInt32:
                    return val.type() == typeid(i32);
                case ShaderCompile::EShaderReflDescriptors::DataUint32:
                    return val.type() == typeid(u32);
                case ShaderCompile::EShaderReflDescriptors::DataInt64:
                    return val.type() == typeid(i64);
                case ShaderCompile::EShaderReflDescriptors::DataUint64:
                    return val.type() == typeid(u64);
                case ShaderCompile::EShaderReflDescriptors::DataFloat:
                    return val.type() == typeid(f32);
                case ShaderCompile::EShaderReflDescriptors::DataDouble:
                    return val.type() == typeid(f64);
                case ShaderCompile::EShaderReflDescriptors::DataVec2:
                    return val.type() == typeid(Vector2f);
                case ShaderCompile::EShaderReflDescriptors::DataVec3:
                    return val.type() == typeid(Vector3f);
                case ShaderCompile::EShaderReflDescriptors::DataVec4:
                    return val.type() == typeid(Vector4f);
                case ShaderCompile::EShaderReflDescriptors::DataVec2i:
                    return val.type() == typeid(Vector2i);
                case ShaderCompile::EShaderReflDescriptors::DataVec3i:
                    return val.type() == typeid(Vector3i);
                case ShaderCompile::EShaderReflDescriptors::DataVec4i:
                    return val.type() == typeid(Vector4i);
                case ShaderCompile::EShaderReflDescriptors::DataVec2u:
                    return val.type() == typeid(Vector2u);
                case ShaderCompile::EShaderReflDescriptors::DataVec3u:
                    return val.type() == typeid(Vector3u);
                case ShaderCompile::EShaderReflDescriptors::DataVec4u:
                    return val.type() == typeid(Vector4u);
                case ShaderCompile::EShaderReflDescriptors::DataMat4f:
                    return val.type() == typeid(Matrix4x4f);
                case ShaderCompile::EShaderReflDescriptors::DataMat2f:
                    return val.type() == typeid(Matrix2x2f);
                case ShaderCompile::EShaderReflDescriptors::BindlessConstantBuffer:
                case ShaderCompile::EShaderReflDescriptors::BindlessStructuredBuffer:
                case ShaderCompile::EShaderReflDescriptors::BindlessRWStructuredBuffer:
                case ShaderCompile::EShaderReflDescriptors::BindlessTexture:
                case ShaderCompile::EShaderReflDescriptors::BindlessRWTexture:
                case ShaderCompile::EShaderReflDescriptors::BindlessSamplerState:
                    return val.type() == typeid(RhiDescriptorHandle);
                default:
                    return false;
            }
        };

        auto& reflData = mData->mReflection;
        for (auto& [k, v] : params.GetAllParameters())
        {
            auto it = reflData.mBindingNameToIndex.find(k);
            if (it == reflData.mBindingNameToIndex.end())
            {
                IF_LOG_ERROR("VA_ShaderVariant",
                    "Shader parameter `{}` not found in shader variant, please check the parameter name.", k);
                return false;
            }
            auto& binding = reflData.mBindings[it->second];
            if (!checkValidAnyCast(v, binding.mType))
            {
                IF_LOG_ERROR(
                    "VA_ShaderVariant", "Shader parameter `{}` type mismatch, please check the parameter type.", k);
                return false;
            }
        }

        // and check if all fields are provided
        for (auto& binding : reflData.mBindings)
        {
            if (params.GetAllParameters().count(binding.mName) == 0)
            {
                IF_LOG_ERROR("VA_ShaderVariant",
                    "Shader parameter `{}` not provided in shader variant, please check the parameter name.",
                    binding.mName);
                return false;
            }
        }

        return true;
    }

    IFRIT_APIDECL u32 VA_ShaderVariant::GetRefl_PushConstantSize() const
    {
        return mData->mReflection.mPushConstantSize;
    }

    IFRIT_APIDECL VkPipelineShaderStageCreateInfo VA_ShaderVariant::GetShaderStageInfo() const
    {
        return mData->mStageCI;
    }
    IFRIT_APIDECL SizedBuffer VA_ShaderVariant::GetRootConstantData(const RhiShaderParameter& params)
    {
        SizedBuffer ret;
        Vec<u8>     rootConstantData(mData->mReflection.mPushConstantSize);
        for (auto& [name, value] : params.GetAllParameters())
        {
            auto it = mData->mReflection.mBindingNameToIndex.find(name);
            if (it != mData->mReflection.mBindingNameToIndex.end())
            {
                auto& binding = mData->mReflection.mBindings[it->second];
                if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataBool)
                {
                    bool v = std::any_cast<bool>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(bool));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataInt32)
                {
                    i32 v = std::any_cast<i32>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(i32));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataUint32)
                {
                    u32 v = std::any_cast<u32>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(u32));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataInt64)
                {
                    i64 v = std::any_cast<i64>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(i64));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataUint64)
                {
                    u64 v = std::any_cast<u64>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(u64));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataFloat)
                {
                    f32 v = std::any_cast<f32>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(f32));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataDouble)
                {
                    f64 v = std::any_cast<f64>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(f64));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataVec2)
                {
                    Vector2f v = std::any_cast<Vector2f>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Vector2f));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataVec4)
                {
                    Vector4f v = std::any_cast<Vector4f>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Vector4f));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataVec2i)
                {
                    Vector2i v = std::any_cast<Vector2i>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Vector2i));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataVec4i)
                {
                    Vector4i v = std::any_cast<Vector4i>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Vector4i));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataVec2u)
                {
                    Vector2u v = std::any_cast<Vector2u>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Vector2u));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataVec4u)
                {
                    Vector4u v = std::any_cast<Vector4u>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Vector4u));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataMat4f)
                {
                    Matrix4x4f v = std::any_cast<Matrix4x4f>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Matrix4x4f));
                }
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::DataMat2f)
                {
                    Matrix2x2f v = std::any_cast<Matrix2x2f>(value);
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &v, sizeof(Matrix2x2f));
                }
                // Bindless
                else if (binding.mType == ShaderCompile::EShaderReflDescriptors::BindlessTexture
                    || binding.mType == ShaderCompile::EShaderReflDescriptors::BindlessRWTexture
                    || binding.mType == ShaderCompile::EShaderReflDescriptors::BindlessStructuredBuffer
                    || binding.mType == ShaderCompile::EShaderReflDescriptors::BindlessRWStructuredBuffer
                    || binding.mType == ShaderCompile::EShaderReflDescriptors::BindlessConstantBuffer
                    || binding.mType == ShaderCompile::EShaderReflDescriptors::BindlessSamplerState)
                {
                    RhiDescriptorHandle v  = std::any_cast<RhiDescriptorHandle>(value);
                    auto                id = v.GetId();
                    memcpy(rootConstantData.data() + binding.mPushConstantOffset, &id, sizeof(u32));
                }
            }
        }
        ret = SizedBuffer(rootConstantData.data(),(rootConstantData.size()));
        return ret;
    }

    // ===== Shader =====

    struct VA_ShaderInternal
    {
        VA_Device*                          mDevice = nullptr;
        Vec<String>                         mDefineNames;
        HashMap<String, u32>                mDefineIds;
        HashMap<u64, Ref<VA_ShaderVariant>> mShaderVariants;
        Vec<u32>                            mMultiCompileIds;
        bool                                mMultiCompileReady = false;
        String                              mPreprocessedCode;
    };

    VA_Shader::VA_Shader(VA_Device* device, const RhiShaderCreateDesc& desc) : RhiShader(desc)
    {
        mContext       = device;
        mData          = new VA_ShaderInternal();
        mData->mDevice = device;

        // Create shader
        auto        content = ReadTextFile(desc.mFilePath);
        auto        defines = SplitString(content, "\n");
        Vec<String> glslLines;

        for (const auto& define : defines)
        {
            if (define.starts_with("#pragma"))
            {
                auto tokens = SplitString(define, " ");
                if (tokens[1] == "ifrit.multi_compile")
                {
                    auto defineName = tokens[2];
                    mData->mDefineNames.push_back(defineName);
                    mData->mDefineIds[defineName] = SizeCast<u32>(mData->mDefineNames.size()) - 1;
                    mData->mMultiCompileIds.push_back(SizeCast<u32>(mData->mDefineNames.size()) - 1);
                }
                else if (tokens[1] == "ifrit.shader_feature")
                {
                    auto defineName = tokens[2];
                    mData->mDefineNames.push_back(defineName);
                    mData->mDefineIds[defineName] = SizeCast<u32>(mData->mDefineNames.size()) - 1;
                }
            }
            else
            {
                glslLines.push_back(define);
            }
        }
        String glslCode          = JoinString(glslLines, "\n");
        mData->mPreprocessedCode = glslCode;
    }
    VA_Shader::~VA_Shader()
    {
        IF_LOG_DEBUG("VA_Shader", "Destroying shader: {}", mDesc.mName);
        for (auto& [k, v] : mData->mShaderVariants)
        {
            v = nullptr;
        }
        mData->mShaderVariants.clear();
        delete mData;
    }

    IFRIT_APIDECL void VA_Shader::PrecompileMultiCompileShaders()
    {
        auto maxMultiCompiles = RHI::GetMaxMultiCompileDirectives();
        if (mData->mMultiCompileIds.size() >= maxMultiCompiles)
        {
            IF_LOG_ERROR("VA_Shader",
                "Multi compile shaders are limited to {} permutations, please reduce the number of permutations.",
                maxMultiCompiles);
            return;
        }
        PrecompileMultiCompileShadersImpl(0, 0);
        mData->mMultiCompileReady = true;
    }
    IFRIT_APIDECL void VA_Shader::PrecompileMultiCompileShadersImpl(u32 curVariantTag, u64 curPermId)
    {
        if (curVariantTag == mData->mMultiCompileIds.size())
        {
            CompileShaderVariant(curPermId);
            return;
        }
        u64 retainedId = curPermId;
        PrecompileMultiCompileShadersImpl(curVariantTag + 1, curPermId);
        retainedId |= (1ull << mData->mMultiCompileIds[curVariantTag]);
        PrecompileMultiCompileShadersImpl(curVariantTag + 1, retainedId);
    }
    IFRIT_APIDECL RHI::RhiShaderVariant* VA_Shader::GetVariant(const Vec<String>& defines)
    {
        u64 permId = 0;
        for (const auto& define : defines)
        {
            if (mData->mDefineIds.count(define) > 0)
            {
                auto id = mData->mDefineIds[define];
                permId |= (1ull << id);
            }
            else
            {
                IF_LOG_ERROR(
                    "VA_Shader", "Shader define `{}` not found in shader collection `{}`", define, mDesc.mName);
                std::abort();
            }
        }
        CompileShaderVariant(permId);
        if (mData->mShaderVariants.count(permId) > 0)
        {
            auto ret = mData->mShaderVariants[permId].get();
            return ret;
        }
        else
        {
            IF_LOG_ERROR("VA_Shader", "Shader variant `{}` not found in shader collection `{}`", permId, mDesc.mName);
            std::abort();
        }
    }
    IFRIT_APIDECL bool VA_Shader::IsMultiCompileReady() { return mData->mMultiCompileReady; }

    IFRIT_APIDECL void VA_Shader::CompileShaderVariant(u64 permId)
    {
        if (mData->mShaderVariants.count(permId) > 0)
        {
            return;
        }

        Vec<String> defines;
        auto        permIdCopy = permId;
        while (permId)
        {
            u32 trailingBit = Math::CountTrailingZero(permId);
            permId &= ~(1 << trailingBit);
            defines.push_back(mData->mDefineNames[trailingBit]);
        }

        auto stageTranslate = [](RHI::ERhiShaderStage stage) -> ShaderCompile::EShaderCompileStage {
            switch (stage)
            {
                case RHI::ERhiShaderStage::Vertex:
                    return ShaderCompile::EShaderCompileStage::VertexShader;
                case RHI::ERhiShaderStage::Fragment:
                    return ShaderCompile::EShaderCompileStage::FragmentShader;
                case RHI::ERhiShaderStage::Compute:
                    return ShaderCompile::EShaderCompileStage::ComputeShader;
                case RHI::ERhiShaderStage::Mesh:
                    return ShaderCompile::EShaderCompileStage::MeshShader;
                case RHI::ERhiShaderStage::Task:
                    return ShaderCompile::EShaderCompileStage::AmplificationShader;
                default:
                    IF_LOG_CRITICAL("VA_Shader", "Unsupported shader stage: `{}`", static_cast<u32>(stage));
                    return ShaderCompile::EShaderCompileStage::VertexShader; // Fallback
            }
        };

        auto sourceTypeConvert = [](RHI::ERhiShaderSourceType sourceType) -> ShaderCompile::EShaderSourceFormat {
            switch (sourceType)
            {
                case RHI::ERhiShaderSourceType::GLSLCode:
                    return ShaderCompile::EShaderSourceFormat::GLSL;
                case RHI::ERhiShaderSourceType::SlangCode:
                    return ShaderCompile::EShaderSourceFormat::Slang;
                case RHI::ERhiShaderSourceType::HLSLCode:
                    return ShaderCompile::EShaderSourceFormat::HLSL;
                default:
                    IF_LOG_CRITICAL("Shader", "Unsupported shader source type: `{}`", static_cast<u32>(sourceType));
                    return ShaderCompile::EShaderSourceFormat::GLSL; // Fallback
            }
        };

        ShaderCompile::ShaderCompileJob job;
        HashMap<String, String>         definitionsInternal;
        job.mName           = mDesc.mName;
        job.mSource.mCode   = String(mData->mPreprocessedCode.begin(), mData->mPreprocessedCode.end());
        job.mSource.mFormat = sourceTypeConvert(mDesc.mSourceType);
        job.mEntryPoint     = mDesc.mEntryPoint;
        job.mStage          = stageTranslate(mDesc.mStage);
        for (const auto& define : defines)
        {
            definitionsInternal[define] = "1"; // Set all defines to 1
        }
        job.mDefinitions = definitionsInternal;

        auto compiler = ShaderCompile::ShaderCompileHelper();
        if (job.mSource.mFormat == ShaderCompile::EShaderSourceFormat::Slang)
        {
            compiler.SetIncludeBase(IFRIT_VKRHI2_SHARED_SHADER_NEXT_INCLUDE_BASE);
        }
        else
        {
            compiler.SetIncludeBase(IFRIT_VKRHI2_SHARED_SHADER_PATH);
        }

        compiler.SetCacheDir(mData->mDevice->GetCacheDir());
        compiler.SetOptimization(ShaderCompile::EShaderCompileOptimization::Performance);

        auto               output = compiler.CompileShaderFromSource(job, ShaderCompile::EShaderIRFormat::SpirV);
        auto               irSize = output.mIR.mData.GetSize();

        VA_ShaderVariantCI shaderModuleCI;
        shaderModuleCI.mIRCode     = output.mIR.mData.ToString();
        shaderModuleCI.mEntryPoint = "main"; // m_CI.m_EntryPoint;
        shaderModuleCI.mStage      = mDesc.mStage;
        shaderModuleCI.mShaderName = mDesc.mName;
        if (defines.size() > 0)
        {
            shaderModuleCI.mShaderName += "(" + JoinString(defines, ".") + ")";
        }

        auto reflectionData = output.mReflData;

        auto shaderModule = MakeOwner<VA_ShaderVariant>(mData->mDevice, shaderModuleCI, &reflectionData);
        mData->mShaderVariants[permIdCopy] = std::move(shaderModule);
    }

} // namespace Ifrit::RHI::VulkanRHI2