#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/ShaderRuntime.h"

namespace Ifrit::Engine::ComLLVMRuntime {
	struct WrappedLLVMRuntimeContext;
	class WrappedLLVMRuntime : public ShaderRuntime {
	public:
		WrappedLLVMRuntime();
		~WrappedLLVMRuntime();
		static void initLlvmBackend();
		virtual void loadIR(std::string irCode, std::string irIdentifier) override;
		virtual void* lookupSymbol(std::string symbol) override;
		virtual std::unique_ptr<ShaderRuntime> getThreadLocalCopy() override;
	private:
		WrappedLLVMRuntimeContext* session;
	};
}