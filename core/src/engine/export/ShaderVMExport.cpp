#include "./engine/export/ShaderVMExport.h"
#include "./engine/shadervm/spirv/SpvVMInterpreter.h"
#include "./engine/shadervm/spirv/SpvVMReader.h"
#include "./engine/shadervm/spirv/SpvVMShader.h"
#include "./engine/base/ShaderRuntime.h"

using namespace Ifrit::Engine::ShaderVM::Spirv;
using namespace Ifrit::Engine;

IFRIT_APIDECL_COMPAT void* IFRIT_APICALL ifspvmCreateVertexShaderFromFile(void* runtime, const char* path) IFRIT_EXPORT_COMPAT_NOTHROW {
	SpvVMReader reader;
	auto fsCode = reader.readFile(path);
	return new SpvVertexShader(*(ShaderRuntimeBuilder*)runtime, fsCode);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifspvmDestroyVertexShaderFromFile(void* p) IFRIT_EXPORT_COMPAT_NOTHROW {
	delete p;
}

IFRIT_APIDECL_COMPAT void* IFRIT_APICALL ifspvmCreateFragmentShaderFromFile(void* runtime, const char* path) IFRIT_EXPORT_COMPAT_NOTHROW {
	SpvVMReader reader;
	auto fsCode = reader.readFile(path);
	return new SpvFragmentShader(*(ShaderRuntimeBuilder*)runtime, fsCode);
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifspvmDestroyFragmentShaderFromFile(void* p) IFRIT_EXPORT_COMPAT_NOTHROW {
	delete p;
}