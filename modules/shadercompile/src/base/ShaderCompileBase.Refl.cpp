#include "ifrit/shadercompile/base/ShaderCompileBase.h"
#include "ifrit/core/reflection/Archive.h"

namespace Ifrit::ShaderCompile
{
    IFRIT_APIDECL void ShaderBinding::DoSerialize(Reflection::Archive* archive) const
    {
        archive->BeginObject("__ifrit_shader_binding");

        archive->BeginObject("__ifrit_name");
        archive->Serialize(mName);
        archive->EndObject();

        archive->BeginObject("__ifrit_type");
        archive->Serialize(static_cast<u32>(mType));
        archive->EndObject();

        archive->BeginObject("__ifrit_set");
        archive->Serialize(mSet);
        archive->EndObject();

        archive->BeginObject("__ifrit_binding");
        archive->Serialize(mBinding);
        archive->EndObject();

        archive->BeginObject("__ifrit_array_size");
        archive->Serialize(mArraySize);
        archive->EndObject();

        archive->BeginObject("__ifrit_push_constant_offset");
        archive->Serialize(mPushConstantOffset);
        archive->EndObject();

        archive->EndObject();
    }

    IFRIT_APIDECL void ShaderBinding::DoDeserialize(Reflection::Archive* archive)
    {
        archive->BeginObject("__ifrit_shader_binding");

        archive->BeginObject("__ifrit_name");
        archive->Serialize(mName);
        archive->EndObject();

        u32 type = 0;
        archive->BeginObject("__ifrit_type");
        archive->Serialize(type);
        mType = static_cast<EShaderReflDescriptors>(type);
        archive->EndObject();

        archive->BeginObject("__ifrit_set");
        archive->Serialize(mSet);
        archive->EndObject();

        archive->BeginObject("__ifrit_binding");
        archive->Serialize(mBinding);
        archive->EndObject();

        archive->BeginObject("__ifrit_array_size");
        archive->Serialize(mArraySize);
        archive->EndObject();

        archive->BeginObject("__ifrit_push_constant_offset");
        archive->Serialize(mPushConstantOffset);
        archive->EndObject();

        archive->EndObject();
    }

    IFRIT_APIDECL void ShaderReflectionData::DoSerialize(Reflection::Archive* archive) const
    {
        archive->BeginObject("__ifrit_shader_reflection_data");

        archive->BeginObject("__ifrit_bindings");
        archive->BeginArray("__ifrit_bindings_array");
        for (const auto& binding : mBindings)
        {
            binding.DoSerialize(archive);
        }
        archive->EndArray();
        archive->EndObject();

        archive->BeginObject("__ifrit_push_constant_size");
        archive->Serialize(mPushConstantSize);
        archive->EndObject();

        archive->BeginObject("__ifrit_valid");
        archive->Serialize(mValid);
        archive->EndObject();

        archive->EndObject();
    }

    IFRIT_APIDECL void ShaderReflectionData::DoDeserialize(Reflection::Archive* archive)
    {
        archive->BeginObject("__ifrit_shader_reflection_data");

        if (archive->HasObject("__ifrit_bindings"))
        {
            archive->BeginObject("__ifrit_bindings");
            if (archive->HasArray("__ifrit_bindings_array"))
            {
                archive->BeginArray("__ifrit_bindings_array");
                mBindings.clear();
                mBindingNameToIndex.clear();
                archive->BeginArrayIteration();
                while (archive->HasNextArrayElement())
                {
                    archive->NextArrayElement();
                    ShaderBinding binding;
                    binding.DoDeserialize(archive);
                    u32 index = static_cast<u32>(mBindings.size());
                    mBindings.push_back(binding);
                    mBindingNameToIndex[binding.mName] = index;
                }
                archive->EndArray();
            }
            archive->EndObject();
        }

        archive->BeginObject("__ifrit_push_constant_size");
        archive->Serialize(mPushConstantSize);
        archive->EndObject();

        archive->BeginObject("__ifrit_valid");
        archive->Serialize(mValid);
        archive->EndObject();

        archive->EndObject();
    }

} // namespace Ifrit::ShaderCompile