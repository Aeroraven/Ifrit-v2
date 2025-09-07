
#pragma once
#include "ifrit/shadercompile/base/ShaderCompileBase.h"

namespace Ifrit::ShaderCompile
{

    enum class EShaderBindlessParamRegResult : u8
    {
        Invalid,
        Success,
        EntryConflict,
    };

    class IFRIT_SHADERCOMPILE_API ShaderReflectionHelper
    {
    public:
        void                          ResetReflectionData();
        ShaderReflectionData          GetReflectionData();

        bool                          IsGlobalParameterPushConstant(StringView name) const;
        EShaderBindlessParamRegResult RegisterArithmeticShaderParams(
            const String& name, EShaderScalarType type, u32 count, bool isMatrix, u32 offset);
        EShaderBindlessParamRegResult RegisterBindlessHandle(const String& name, StringView tpname, u32 offset);
        inline void                   SetRootConstantSize(u32 size) { mReflData.mPushConstantSize = size; }

    private:
        ShaderReflectionData mReflData{};
    };
} // namespace Ifrit::ShaderCompile