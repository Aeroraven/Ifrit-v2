#include "engine/shadervm/spirv/SpvVMShader.h"

namespace Ifrit::Engine::ShaderVM::Spirv {
	int SpvRuntimeBackend::createTime = 0;
	SpvRuntimeBackend::SpvRuntimeBackend(ShaderRuntime* runtime, std::vector<char> irByteCode) {
		reader.initializeContext(&spctx);
		reader.parseByteCode(irByteCode.data(), irByteCode.size() / 4, &spctx);
		interpeter.parseRawContext(&spctx, &spvir);
		this->runtime = runtime;
		interpeter.exportLlvmIR(&spvir, &this->irCode);
		runtime->loadIR(this->irCode, std::to_string(this->createTime));
		updateSymbolTable(false);
	}
	SpvRuntimeBackend::SpvRuntimeBackend(const SpvRuntimeBackend& other) {
		this->copiedRuntime = other.runtime->getThreadLocalCopy();
		this->runtime = this->copiedRuntime.get();
		this->irCode = other.irCode;
		this->spvirRef = &other.spvir;
		updateSymbolTable(true);
	}
	void SpvRuntimeBackend::updateSymbolTable(bool isCopy) {
		const SpvVMIntermediateRepresentation* svmir = (isCopy) ? spvirRef: &spvir;

		this->symbolTables.inputs.resize(svmir->shaderMaps.inputVarSymbols.size());
		this->symbolTables.outputs.resize(svmir->shaderMaps.outputVarSymbols.size());
		this->symbolTables.inputBytes.resize(svmir->shaderMaps.inputVarSymbols.size());
		this->symbolTables.outputBytes.resize(svmir->shaderMaps.outputVarSymbols.size());
		for (int i = 0; i < svmir->shaderMaps.inputVarSymbols.size(); i++) {
			this->symbolTables.inputs[i] = this->runtime->lookupSymbol(svmir->shaderMaps.inputVarSymbols[i]);
			this->symbolTables.inputBytes[i] = svmir->shaderMaps.inputSize[i];
		}
		for (int i = 0; i < svmir->shaderMaps.outputVarSymbols.size(); i++) {
			this->symbolTables.outputs[i] = this->runtime->lookupSymbol(svmir->shaderMaps.outputVarSymbols[i]);
			this->symbolTables.outputBytes[i] = svmir->shaderMaps.outputSize[i];
		}
		for (int i = 0; i < svmir->shaderMaps.uniformSize.size(); i++) {
			std::pair<int, int> loc = svmir->shaderMaps.uniformVarLoc[i];
			this->symbolTables.uniform[loc] = {
				this->runtime->lookupSymbol(svmir->shaderMaps.uniformVarSymbols[i]),
				svmir->shaderMaps.uniformSize[i]
			};
		}
		this->symbolTables.entry = this->runtime->lookupSymbol(svmir->shaderMaps.mainFuncSymbol);
		if (svmir->shaderMaps.builtinPositionSymbol.size()) {
			this->symbolTables.builtinPosition = this->runtime->lookupSymbol(svmir->shaderMaps.builtinPositionSymbol);
		}
	}

	SpvVertexShader::SpvVertexShader(ShaderRuntime* runtime, std::vector<char> irByteCode):SpvRuntimeBackend(runtime, irByteCode){
		isThreadSafe = false;
	}
	SpvVertexShader::SpvVertexShader(const SpvVertexShader& p) :SpvRuntimeBackend(p) {
		isThreadSafe = false;
	}
	IFRIT_HOST VertexShader* SpvVertexShader::getThreadLocalCopy() {
		auto copy = new SpvVertexShader(*this);
		return copy;
	}

	IFRIT_HOST void SpvVertexShader::updateUniformData(int binding, int set, const void* pData){
		auto& uniformData = symbolTables.uniform[{binding, set}];
		memcpy(uniformData.first, pData, uniformData.second);
	}

	IFRIT_HOST std::vector<std::pair<int, int>> SpvVertexShader::getUniformList(){
		std::vector<std::pair<int, int>> ret;
		for (auto& p : this->symbolTables.uniform) {
			ret.push_back(p.first);
		}
		return ret;
	}

	IFRIT_HOST VaryingDescriptor SpvVertexShader::getVaryingDescriptor(){
		VaryingDescriptor vdesc;
		std::vector<TypeDescriptor> tpDesc{};
		for (int i = 0; i < symbolTables.outputs.size(); i++) {
			tpDesc.push_back(TypeDescriptors.FLOAT4);
		}
		vdesc.setVaryingDescriptors(tpDesc);
		return vdesc;
	}
	
	SpvFragmentShader::SpvFragmentShader(ShaderRuntime* runtime, std::vector<char> irByteCode) :SpvRuntimeBackend(runtime, irByteCode) {
		isThreadSafe = false;
	}
	SpvFragmentShader::SpvFragmentShader(const SpvFragmentShader& p) :SpvRuntimeBackend(p) {
		isThreadSafe = false;
	}
	IFRIT_HOST FragmentShader* SpvFragmentShader::getThreadLocalCopy() {
		auto copy = new SpvFragmentShader(*this);
		return copy;
	}

	IFRIT_HOST void SpvFragmentShader::updateUniformData(int binding, int set, const void* pData){
		auto& uniformData = symbolTables.uniform[{binding, set}];
		memcpy(uniformData.first, pData, uniformData.second);
	}

	void SpvVertexShader::execute(const void* const* input, ifloat4* outPos, VaryingStore* const* outVaryings) {
		//TODO: Input & Output
		for (int i = 0; i < symbolTables.inputBytes.size(); i++) {
			memcpy(symbolTables.inputs[i], input[i], symbolTables.inputBytes[i]);
		}
		auto shaderEntry = (void(*)())this->symbolTables.entry;
		shaderEntry();
		for (int i = 0; i < symbolTables.outputs.size(); i++) {
			memcpy(outVaryings[i], symbolTables.outputs[i], symbolTables.outputBytes[i]);
		}
		if(symbolTables.builtinPosition) memcpy(outPos, symbolTables.builtinPosition, 16);

	}
	IFRIT_HOST VertexShader* SpvVertexShader::getCudaClone(){
		ifritError("CUDA not supported");
		return nullptr;
	}
	void SpvFragmentShader::execute(const void* varyings, void* colorOutput, float* fragmentDepth) {
		//TODO: Input & Output
		for (int i = 0; i < symbolTables.inputBytes.size(); i++) {
			memcpy(symbolTables.inputs[i], (VaryingStore*)varyings + i, symbolTables.inputBytes[i]);
		}
		auto shaderEntry = (void(*)())this->symbolTables.entry;
		shaderEntry();
		for (int i = 0; i < symbolTables.outputs.size(); i++) {
			memcpy((ifloat4*)colorOutput + i, symbolTables.outputs[i], symbolTables.outputBytes[i]);
		}
	}
	IFRIT_HOST FragmentShader* SpvFragmentShader::getCudaClone(){
		ifritError("CUDA not supported");
		return nullptr;
	}
	
	IFRIT_HOST std::vector<std::pair<int, int>> SpvFragmentShader::getUniformList() {
		std::vector<std::pair<int, int>> ret;
		for (auto& p : this->symbolTables.uniform) {
			ret.push_back(p.first);
		}
		return ret;
	}
}