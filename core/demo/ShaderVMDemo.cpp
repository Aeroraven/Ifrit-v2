#include "engine/shadervm/spirv/SpvVMReader.h"
#include "engine/shadervm/spirv/SpvVMInterpreter.h"
#include "ShaderVMDemo.h"

using namespace Ifrit::Engine::ShaderVM::Spirv;

namespace Ifrit::Demo::ShaderVMDemo {
	void llvmJITHelloWorld() {
		
	}
	int mainEntry() {
		llvmJITHelloWorld();
		/*
		SpvVMReader reader;
		SpvVMContext context;
		SpvVMInterpreter interpreter;
		SpvVMIntermediateRepresentation irctx;
		std::string outLlvmIr;
		auto bytecode = reader.readFile( R"(C:\WR\Aria\Anthem\shader\glsl\ssaoComposition\shader.frag.spv)" );
		reader.initializeContext(&context);
		reader.parseByteCode(bytecode.data(), bytecode.size() / 4, &context);
		reader.printParsedInstructions(&context);
		interpreter.parseRawContext(&context, &irctx);

		interpreter.exportLlvmIR(&irctx, &outLlvmIr);*/

		return 0;
	}
}