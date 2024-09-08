#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/Shaders.h"
#include "./engine/base/ShaderRuntime.h"
#include "./engine/shadervm/spirv/SpvVMInterpreter.h"
#include "./engine/shadervm/spirv/SpvVMReader.h"

namespace Ifrit::Engine::ShaderVM::Spirv {
	struct SpvRuntimeSymbolTables {
		std::vector<void*> inputs;
		std::vector<int> inputBytes;
		std::vector<void*> outputs;
		std::vector<int> outputBytes;
		std::unordered_map<std::pair<int, int>, std::pair<void*, int>,Ifrit::Core::Utility::PairHash> uniform;
		void* entry = nullptr;
		void* builtinPosition = nullptr;
	};
	class SpvRuntimeBackend {
	protected:
		static int createTime;
		SpvVMReader reader;
		SpvVMInterpreter interpeter;
		SpvVMContext spctx;
		SpvVMIntermediateRepresentation spvir;
		const SpvVMIntermediateRepresentation* spvirRef;
		ShaderRuntime* runtime;
		SpvRuntimeSymbolTables symbolTables;
		std::unique_ptr<ShaderRuntime> copiedRuntime = nullptr;
		std::string irCode;
		
		std::unique_ptr<ShaderRuntime> owningRuntime = nullptr;
	public:
		SpvRuntimeBackend(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
		SpvRuntimeBackend(const SpvRuntimeBackend& other);
	protected:
		void updateSymbolTable(bool isCopy);
	};

	class SpvVertexShader final: public VertexShader, public SpvRuntimeBackend {
	public:
		SpvVertexShader(const SpvVertexShader& p);
	public:
		SpvVertexShader(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
		~SpvVertexShader() = default;
		IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, VaryingStore* const* outVaryings) override;
		IFRIT_HOST virtual VertexShader* getCudaClone() override;
		IFRIT_HOST virtual std::unique_ptr<VertexShader> getThreadLocalCopy() override;
		IFRIT_HOST virtual void updateUniformData(int binding, int set, const void* pData) override;
		IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
		IFRIT_HOST virtual VaryingDescriptor getVaryingDescriptor() override;
	};

	class SpvFragmentShader final: public FragmentShader, public SpvRuntimeBackend {
	public:
		SpvFragmentShader(const SpvFragmentShader& p);
	public:
		SpvFragmentShader(const ShaderRuntimeBuilder& runtime, std::vector<char> irByteCode);
		~SpvFragmentShader() = default;
		IFRIT_DUAL virtual void execute(const void* varyings,void* colorOutput,	float* fragmentDepth) override;
		IFRIT_HOST virtual FragmentShader* getCudaClone() override;
		IFRIT_HOST virtual std::unique_ptr<FragmentShader> getThreadLocalCopy() override;
		IFRIT_HOST virtual void updateUniformData(int binding, int set, const void* pData) override;
		IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() override;
	};
}