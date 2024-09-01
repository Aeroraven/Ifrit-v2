#include "engine/shadervm/spirv/SpvVMReader.h"
#include "engine/shadervm/spirv/SpvVMInterpreter.h"
#include "ShaderVMDemo.h"
#include "engine/comllvmrt/WrappedLLVMRuntime.h"

using namespace Ifrit::Engine::ShaderVM::Spirv;
using namespace Ifrit::Engine::ComLLVMRuntime;

namespace Ifrit::Demo::ShaderVMDemo {
	void llvmJITHelloWorld() {
		
	}
	int mainEntry() {
		
		SpvVMReader reader;
		SpvVMContext context;
		SpvVMInterpreter interpreter;
		SpvVMIntermediateRepresentation irctx;
		std::string outLlvmIr;
		auto bytecode = reader.readFile( R"(C:\WR\Aria\Anthem\shader\glsl\tex3d\shader.first.frag.spv)" );
		reader.initializeContext(&context);
		reader.parseByteCode(bytecode.data(), bytecode.size() / 4, &context);
		reader.printParsedInstructions(&context);
		interpreter.parseRawContext(&context, &irctx);
		interpreter.exportLlvmIR(&irctx, &outLlvmIr);

		WrappedLLVMRuntime irt;
		irt.initLlvmBackend();
		irt.loadIR(outLlvmIr, "Test");
		
		return 0;
	}
}