#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/common/Pch.h"

#ifdef __INTELLISENSE__
    #define IFRIT_RDG_API
#else
    #define IFRIT_RDG_API IFRIT_RUNTIME_API
#endif

namespace Ifrit::Runtime::RDG
{
    enum class EShaderParamDescType
    {
        Undefined    = 0,
        PushConstant = 1,
    };

    class IFRIT_RDG_API ShaderParamsDesc
    {
    public:
        void Test();

    private:
        EShaderParamDescType m_Type = EShaderParamDescType::Undefined;
    };
} // namespace Ifrit::Runtime::RDG