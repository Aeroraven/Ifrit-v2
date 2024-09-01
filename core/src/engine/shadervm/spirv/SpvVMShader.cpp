#include "engine/shadervm/spirv/SpvVMShader.h"

namespace Ifrit::Engine::ShaderVM::Spirv {
	int SpvRuntimeBackend::createTime = 0;
	SpvRuntimeBackend::SpvRuntimeBackend(ShaderRuntime* runtime, std::string irByteCode) {
		reader.initializeContext(&spctx);
		reader.parseByteCode(irByteCode.c_str(), irByteCode.size() / 4, &spctx);
		interpeter.parseRawContext(&spctx, &spvir);
		this->runtime = runtime;
		interpeter.exportLlvmIR(&spvir, &this->irCode);
		runtime->loadIR(this->irCode, std::to_string(this->createTime));
		updateSymbolTable();
	}
	SpvRuntimeBackend::SpvRuntimeBackend(const SpvRuntimeBackend& other) {
		this->copiedRuntime = other.runtime->getThreadLocalCopy();
		this->runtime = this->copiedRuntime.get();
		updateSymbolTable();
	}
	void SpvRuntimeBackend::updateSymbolTable() {
		this->symbolTables.inputs.resize(this->spvir.shaderMaps.inputVarSymbols.size());
		this->symbolTables.outputs.resize(this->spvir.shaderMaps.outputVarSymbols.size());
		for (int i = 0; i < this->spvir.shaderMaps.inputVarSymbols.size(); i++) {
			this->symbolTables.inputs[i] = this->runtime->lookupSymbol(this->spvir.shaderMaps.inputVarSymbols[i]);
		}
		for (int i = 0; i < this->spvir.shaderMaps.outputVarSymbols.size(); i++) {
			this->symbolTables.outputs[i] = this->runtime->lookupSymbol(this->spvir.shaderMaps.outputVarSymbols[i]);
		}
		this->symbolTables.entry = this->runtime->lookupSymbol(this->spvir.shaderMaps.mainFuncSymbol);
	}

	SpvVertexShader::SpvVertexShader(ShaderRuntime* runtime, std::string irByteCode):SpvRuntimeBackend(runtime, irByteCode){}
	SpvVertexShader::SpvVertexShader(const SpvVertexShader& p) :SpvRuntimeBackend(p) {}
	IFRIT_HOST VertexShader* SpvVertexShader::getThreadLocalCopy() {
		auto copy = new SpvVertexShader(*this);
		return copy;
	}
	
	SpvFragmentShader::SpvFragmentShader(ShaderRuntime* runtime, std::string irByteCode) :SpvRuntimeBackend(runtime, irByteCode) {}
	SpvFragmentShader::SpvFragmentShader(const SpvFragmentShader& p) :SpvRuntimeBackend(p) {}
	IFRIT_HOST FragmentShader* SpvFragmentShader::getThreadLocalCopy() {
		auto copy = new SpvFragmentShader(*this);
		return copy;
	}

	void SpvVertexShader::execute(const void* const* input, ifloat4* outPos, VaryingStore* const* outVaryings) {
		//TODO: Input & Output
		auto shaderEntry = (void(*)())this->symbolTables.entry;
		shaderEntry();
	}
	void SpvFragmentShader::execute(const void* varyings, void* colorOutput, float* fragmentDepth) {
		//TODO: Input & Output
		auto shaderEntry = (void(*)())this->symbolTables.entry;
		shaderEntry();
	}
}