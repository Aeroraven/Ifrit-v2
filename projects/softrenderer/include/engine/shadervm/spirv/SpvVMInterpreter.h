#pragma once
#include "./core/definition/CoreExports.h"
#include "./SpvVMContext.h"
namespace Ifrit::Engine::ShaderVM::Spirv {
	class SpvVMInterpreter {
	public:
		void parseRawContext(SpvVMContext* context, SpvVMIntermediateRepresentation* outIr);
		void exportLlvmIR(SpvVMIntermediateRepresentation* ir, std::string* outLlvmIR);
	};
}