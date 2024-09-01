#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/Shaders.h"
#include "./engine/base/ShaderRuntime.h"
#include "./engine/shadervm/spirv/SpvVMInterpreter.h"
#include "./engine/shadervm/spirv/SpvVMReader.h"

namespace Ifrit::Engine::ShaderVM::Spirv {
	struct SpvRuntimeSymbolTables {
		std::vector<void*> inputs;
		std::vector<void*> outputs;
		void* entry;
	};
	class SpvRuntimeBackend {
	protected:
		static int createTime;
		SpvVMReader reader;
		SpvVMInterpreter interpeter;
		SpvVMContext spctx;
		SpvVMIntermediateRepresentation spvir;
		ShaderRuntime* runtime;
		SpvRuntimeSymbolTables symbolTables;
		std::unique_ptr<ShaderRuntime> copiedRuntime = nullptr;
		std::string irCode;
		
	public:
		SpvRuntimeBackend(ShaderRuntime* runtime, std::string irByteCode);
		SpvRuntimeBackend(const SpvRuntimeBackend& other);
	protected:
		void updateSymbolTable();
	};

	class SpvVertexShader : public VertexShader, public SpvRuntimeBackend {
	protected:
		SpvVertexShader(const SpvVertexShader& p);
	public:
		SpvVertexShader(ShaderRuntime* runtime, std::string irByteCode);
		~SpvVertexShader() = default;
		IFRIT_DUAL virtual void execute(const void* const* input, ifloat4* outPos, VaryingStore* const* outVaryings) override;
		IFRIT_HOST virtual VertexShader* getCudaClone() override;
		IFRIT_HOST virtual VertexShader* getThreadLocalCopy() override;
	}

	class SpvFragmentShader : public FragmentShader, public SpvRuntimeBackend {
	protected:
		SpvFragmentShader(const SpvFragmentShader& p);
	public:
		SpvFragmentShader(ShaderRuntime* runtime, std::string irByteCode);
		~SpvFragmentShader() = default;
		IFRIT_DUAL virtual void execute(const void* varyings,void* colorOutput,	float* fragmentDepth) override;
		IFRIT_HOST virtual FragmentShader* getCudaClone() override;
		IFRIT_HOST virtual FragmentShader* getThreadLocalCopy() override;
	};
}