#include "ifrit/shadercompile/base/ShaderCompileBase.h"

namespace Ifrit::ShaderCompile
{

    IFRIT_APIDECL void ShaderCompilerBase::SetCachePath(const String& cacheDir) { mCachePath = cacheDir; }

    IFRIT_APIDECL void ShaderCompilerBase::SetIncludeBase(const String& includeBase) { mIncludeBase = includeBase; }

    IFRIT_APIDECL void ShaderCompilerBase::SetOptimization(EShaderCompileOptimization optimization)
    {
        mOptimization = optimization;
    }

} // namespace Ifrit::ShaderCompile