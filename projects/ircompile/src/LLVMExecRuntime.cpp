#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <memory>

// Mem2reg
#include "ifrit/ircompile/llvm_ircompile.h"
#include "llvm/Transforms/Utils.h"

using namespace llvm;
using namespace llvm::orc;

ExitOnError ExitOnErr;

static ThreadSafeModule optimizeModule(ThreadSafeModule M) {
  // Create a function pass manager.
  auto FPM =
      std::make_unique<legacy::FunctionPassManager>(M.getModuleUnlocked());

  // Add some optimizations.
  FPM->add(createPromoteMemoryToRegisterPass());
  FPM->add(createInstructionCombiningPass());
  FPM->add(createReassociatePass());
  FPM->add(createGVNPass());
  FPM->add(createCFGSimplificationPass());
  FPM->add(createDeadStoreEliminationPass());
  FPM->add(createDeadCodeEliminationPass());
  FPM->doInitialization();

  // Run the optimizations over all functions in the module being added to
  // the JIT.
  for (auto &F : *M.getModuleUnlocked())
    FPM->run(F);

  // Print IR
  // M.getModuleUnlocked()->print(errs(), nullptr);

  return M;
}

struct IfritCompLLVMExecutionSession {
  std::unique_ptr<legacy::PassManager> PM =
      std::make_unique<legacy::PassManager>();

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
    tsModule =
        optimizeModule(ThreadSafeModule(std::move(M), std::move(llvmCtx)));
    ExitOnErr(jit->addIRModule(std::move(tsModule)));
    ready = true;
  }
  void *lookupSymbol(std::string symbol) {
    if (!ready) {
      return nullptr;
    }
    auto sym = ExitOnErr(jit->lookup(symbol));
    return (void *)sym.getAddress();
  }
};

IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Init() {
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
}

IFRIT_COM_LE_API IfritCompLLVMExecutionSession *IFRIT_COM_LE_API_CALLCONV
IfritCom_LlvmExec_Create(const char *ir, const char *identifier) {
  auto session = new IfritCompLLVMExecutionSession();
  session->loadIR(ir);
  return session;
}

IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV
IfritCom_LlvmExec_Destroy(IfritCompLLVMExecutionSession *session) {
  delete session;
}

IFRIT_COM_LE_API void *IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Lookup(
    IfritCompLLVMExecutionSession *session, const char *symbol) {
  return session->lookupSymbol(symbol);
}
