
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#ifdef _MSC_VER
#define IFRIT_IGNORE_IRCOMPILE
#endif

#ifndef IFRIT_IGNORE_IRCOMPILE
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
#endif
#include <memory>

// Mem2reg
#include "ifrit/ircompile/llvm_ircompile.h"

#ifndef IFRIT_IGNORE_IRCOMPILE
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
#endif

IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Init() {
#ifndef IFRIT_IGNORE_IRCOMPILE
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
#endif
}

IFRIT_COM_LE_API IfritCompLLVMExecutionSession *IFRIT_COM_LE_API_CALLCONV
IfritCom_LlvmExec_Create(const char *ir, const char *identifier) {
#ifndef IFRIT_IGNORE_IRCOMPILE
  auto session = new IfritCompLLVMExecutionSession();
  session->loadIR(ir);
  return session;
#else
  return nullptr;
#endif
}

IFRIT_COM_LE_API void IFRIT_COM_LE_API_CALLCONV
IfritCom_LlvmExec_Destroy(IfritCompLLVMExecutionSession *session) {
  delete session;
}

IFRIT_COM_LE_API void *IFRIT_COM_LE_API_CALLCONV IfritCom_LlvmExec_Lookup(
    IfritCompLLVMExecutionSession *session, const char *symbol) {
#ifndef IFRIT_IGNORE_IRCOMPILE
  return session->lookupSymbol(symbol);
#else
  return nullptr;
#endif
}
