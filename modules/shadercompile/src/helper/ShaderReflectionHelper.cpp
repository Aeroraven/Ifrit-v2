#include "ifrit/shadercompile/helper/ShaderReflectionHelper.h"
#include "ifrit/core/base/SharedShaderConstants.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/typing/EnumReflection.h"

namespace Ifrit::ShaderCompile
{

    struct BindlessWrapperTypeRegistration
    {
        HashMap<String, EShaderReflDescriptors> mRegisteredNames;

        BindlessWrapperTypeRegistration()
        {
            mRegisteredNames["TRWStructuredBufferHandle"]       = EShaderReflDescriptors::BindlessRWStructuredBuffer;
            mRegisteredNames["TStructuredBufferHandle"]         = EShaderReflDescriptors::BindlessStructuredBuffer;
            mRegisteredNames["TAtomicRWStructuredBufferHandle"] = EShaderReflDescriptors::BindlessRWStructuredBuffer;
            // TODO: remove bindless cbv
            mRegisteredNames["TConstantBufferHandle"]              = EShaderReflDescriptors::BindlessRWStructuredBuffer;
            mRegisteredNames["TRWStructuredBufferHandle_ReadOnly"] = EShaderReflDescriptors::BindlessRWStructuredBuffer;

            mRegisteredNames["TVertexDataHandle"]  = EShaderReflDescriptors::BindlessRWStructuredBuffer;
            mRegisteredNames["TNormalDataHandle"]  = EShaderReflDescriptors::BindlessRWStructuredBuffer;
            mRegisteredNames["TTangentDataHandle"] = EShaderReflDescriptors::BindlessRWStructuredBuffer;
            mRegisteredNames["TUVDataHandle"]      = EShaderReflDescriptors::BindlessRWStructuredBuffer;

            mRegisteredNames["TRWTexture2DHandle"] = EShaderReflDescriptors::BindlessRWTexture;
            mRegisteredNames["TRWTexture3DHandle"] = EShaderReflDescriptors::BindlessRWTexture;
            mRegisteredNames["TTexture2DHandle"]   = EShaderReflDescriptors::BindlessTexture;
            mRegisteredNames["TTexture3DHandle"]   = EShaderReflDescriptors::BindlessTexture;
        }

        EShaderReflDescriptors GetBindlessWrapperType(StringView name)
        {
            // TODO: heterogeneous lookup
            auto it = mRegisteredNames.find(String{ name });
            if (it != mRegisteredNames.end())
            {
                return it->second;
            }
            return EShaderReflDescriptors::Unknown;
        }
    };
    static BindlessWrapperTypeRegistration gBindlessWrapperTypeReg;

    void                                   ShaderReflectionHelper::ResetReflectionData() { mReflData = {}; }

    ShaderReflectionData                   ShaderReflectionHelper::GetReflectionData() { return mReflData; }

    bool                                   ShaderReflectionHelper::IsGlobalParameterPushConstant(StringView name) const
    {
        return name == kIfritShader_PushConstantAuxName;
    }

    EShaderBindlessParamRegResult ShaderReflectionHelper::RegisterArithmeticShaderParams(
        const String& name, EShaderScalarType type, u32 count, bool isMatrix, u32 offset)
    {
        if (mReflData.mBindingNameToIndex.find(name) != mReflData.mBindingNameToIndex.end())
        {
            return EShaderBindlessParamRegResult::EntryConflict;
        }
        EShaderReflDescriptors desiredType;
        switch (type)
        {
            case EShaderScalarType::Float:
                if (isMatrix)
                {
                    if (count == 16)
                        desiredType = EShaderReflDescriptors::DataMat4f;
                    else if (count == 4)
                        desiredType = EShaderReflDescriptors::DataMat2f;
                    else
                        return EShaderBindlessParamRegResult::Invalid;
                }
                else
                {
                    if (count == 1)
                        desiredType = EShaderReflDescriptors::DataFloat;
                    else if (count == 2)
                        desiredType = EShaderReflDescriptors::DataVec2;
                    else if (count == 3)
                        desiredType = EShaderReflDescriptors::DataVec3;
                    else if (count == 4)
                        desiredType = EShaderReflDescriptors::DataVec4;
                    else
                        return EShaderBindlessParamRegResult::Invalid;
                }
                break;
            case EShaderScalarType::Int32:
                if (count == 1)
                    desiredType = EShaderReflDescriptors::DataInt32;
                else if (count == 2)
                    desiredType = EShaderReflDescriptors::DataVec2i;
                else if (count == 3)
                    desiredType = EShaderReflDescriptors::DataVec3i;
                else if (count == 4)
                    desiredType = EShaderReflDescriptors::DataVec4i;
                else
                    return EShaderBindlessParamRegResult::Invalid;
                break;
            case EShaderScalarType::Uint32:
                if (count == 1)
                    desiredType = EShaderReflDescriptors::DataUint32;
                else if (count == 2)
                    desiredType = EShaderReflDescriptors::DataVec2u;
                else if (count == 3)
                    desiredType = EShaderReflDescriptors::DataVec3u;
                else if (count == 4)
                    desiredType = EShaderReflDescriptors::DataVec4u;
                else
                    return EShaderBindlessParamRegResult::Invalid;
                break;
            default:
                return EShaderBindlessParamRegResult::Invalid;
        }
        ShaderBinding binding;
        binding.mName               = name;
        binding.mType               = desiredType;
        binding.mPushConstantOffset = offset;

        u32 index = static_cast<u32>(mReflData.mBindings.size());
        mReflData.mBindings.push_back(binding);
        mReflData.mBindingNameToIndex[name] = index;

        IF_LOG_INFO(
            "ShaderReflectionHelper", "Registered arithmetic param: {}, type: {}", name, GetEnumName(desiredType));

        return EShaderBindlessParamRegResult::Success;
    }

    EShaderBindlessParamRegResult ShaderReflectionHelper::RegisterBindlessHandle(
        const String& name, StringView tpname, u32 offset)
    {
        if (mReflData.mBindingNameToIndex.find(name) != mReflData.mBindingNameToIndex.end())
        {
            return EShaderBindlessParamRegResult::EntryConflict;
        }
        EShaderReflDescriptors desiredType = gBindlessWrapperTypeReg.GetBindlessWrapperType(tpname);
        if (desiredType == EShaderReflDescriptors::Unknown)
        {
            return EShaderBindlessParamRegResult::Invalid;
        }

        ShaderBinding binding;
        binding.mName               = name;
        binding.mType               = desiredType;
        binding.mPushConstantOffset = offset;

        u32 index = static_cast<u32>(mReflData.mBindings.size());
        mReflData.mBindings.push_back(binding);
        mReflData.mBindingNameToIndex[name] = index;

        IF_LOG_INFO(
            "ShaderReflectionHelper", "Registered bindless handle: {}, type: {}", name, GetEnumName(desiredType));
        return EShaderBindlessParamRegResult::Success;
    }
} // namespace Ifrit::ShaderCompile