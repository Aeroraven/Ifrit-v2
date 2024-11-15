#include "ifrit/softgraphics/engine/export/ShaderVMExport.h"
#include "ifrit/softgraphics/engine/base/ShaderRuntime.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMInterpreter.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMReader.h"
#include "ifrit/softgraphics/engine/shadervm/spirv/SpvVMShader.h"

using namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::Spirv;
using namespace Ifrit::GraphicsBackend::SoftGraphics;

IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifspvmCreateVertexShaderFromFile(
    void *runtime, const char *path) IFRIT_EXPORT_COMPAT_NOTHROW {
  SpvVMReader reader;
  auto fsCode = reader.readFile(path);
  return new SpvVertexShader(*(ShaderRuntimeBuilder *)runtime, fsCode);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifspvmDestroyVertexShaderFromFile(void *p) IFRIT_EXPORT_COMPAT_NOTHROW {
  delete (SpvVertexShader *)p;
}

IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifspvmCreateFragmentShaderFromFile(
    void *runtime, const char *path) IFRIT_EXPORT_COMPAT_NOTHROW {
  SpvVMReader reader;
  auto fsCode = reader.readFile(path);
  return new SpvFragmentShader(*(ShaderRuntimeBuilder *)runtime, fsCode);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifspvmDestroyFragmentShaderFromFile(void *p) IFRIT_EXPORT_COMPAT_NOTHROW {
  delete (SpvFragmentShader *)p;
}