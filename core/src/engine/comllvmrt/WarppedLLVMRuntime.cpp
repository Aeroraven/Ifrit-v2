#include "engine/comllvmrt/WrappedLLVMRuntime.h"
#include "LLVMExecRuntime.Decl.h"

namespace Ifrit::Engine::ComLLVMRuntime {
	struct WrappedLLVMRuntimeContext {
		IfritCompLLVMExecutionSession* session;
		std::string irCode;
		std::string irIdentifier;
	};

	WrappedLLVMRuntime::WrappedLLVMRuntime() {
		this->session = new WrappedLLVMRuntimeContext();
		session->session = nullptr;
	}

	WrappedLLVMRuntime::~WrappedLLVMRuntime() {
		if (session) {
			if(session->session)
				IfritCom_LlvmExec_Destroy(session->session);
			delete this->session;
		}
	}

	void WrappedLLVMRuntime::initLlvmBackend(){
		IfritCom_LlvmExec_Init();
	}

	void WrappedLLVMRuntime::loadIR(std::string irCode,std::string irIdentifier) {
		if (session->session) {
			IfritCom_LlvmExec_Destroy(session->session);
		}
		session->irCode	= irCode;
		session->irIdentifier = irIdentifier;
		session->session = IfritCom_LlvmExec_Create(irCode.c_str(), irIdentifier.c_str());
	}

	void* WrappedLLVMRuntime::lookupSymbol(std::string symbol) {
		if (!session->session) {
			return nullptr;
		}
		return IfritCom_LlvmExec_Lookup(session->session, symbol.c_str());
	}

	std::unique_ptr<ShaderRuntime> WrappedLLVMRuntime::getThreadLocalCopy(){
		auto copy = std::make_unique<WrappedLLVMRuntime>();
		copy->loadIR(session->irCode, session->irIdentifier);
		return copy;
	}

}