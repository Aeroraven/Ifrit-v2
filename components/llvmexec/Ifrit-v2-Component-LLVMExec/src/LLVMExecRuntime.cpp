#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include "../include/LLVMExecRuntime.Decl.h"

#ifndef __INTELLISENSE__
    #ifdef _MSC_VER
        static_assert(false, "This project cannot be built using current compiler. For windows, build with MinGW-w64.");
    #endif
#endif

using namespace llvm;
using namespace llvm::orc;

ExitOnError ExitOnErr;

struct IfritCompLLVMExecutionSession {
    std::unique_ptr<LLVMContext> llvmCtx;
    ThreadSafeModule tsModule;
    SMDiagnostic Err;
    LLJITBuilder jitBuilder;
    std::unique_ptr<LLJIT> jit;
    bool ready = false;
    void loadIR(std::string irCode) {
        jit = ExitOnErr(jitBuilder.create());
        jit->getMainJITDylib().addGenerator(
            cantFail(DynamicLibrarySearchGenerator::GetForCurrentProcess(
                jit->getDataLayout().getGlobalPrefix())));
        llvmCtx = std::make_unique<LLVMContext>();
        auto M = parseIR(*MemoryBuffer::getMemBuffer(irCode), Err, *llvmCtx);
        if (!M) {
            Err.print("IfritLLVMExecutionSession: Fail to parse IR:", errs());
            exit(1);
        }
        tsModule = ThreadSafeModule(std::move(M), std::move(llvmCtx));
        ExitOnErr(jit->addIRModule(std::move(tsModule)));
        ready = true;
    }
    void* lookupSymbol(std::string symbol) {
		if (!ready) {
            return nullptr;
		}
		auto sym = ExitOnErr(jit->lookup(symbol));
		return (void*)sym.getAddress();
	}
};

IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Init() {
    InitializeNativeTarget();
	InitializeNativeTargetAsmPrinter();
}

IFRIT_COM_LE_API IfritCompLLVMExecutionSession* IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Create(const char* ir,const char* identifier) {
	auto session = new IfritCompLLVMExecutionSession();
	session->loadIR(ir);
	return session;
}

IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Destroy(IfritCompLLVMExecutionSession* session) {
	delete session;
}

IFRIT_COM_LE_API void* IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Lookup(IfritCompLLVMExecutionSession* session, const char* symbol) {
	return session->lookupSymbol(symbol);
}
