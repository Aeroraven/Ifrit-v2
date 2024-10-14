#include "./engine/export/ComLLVMRtExport.h"
#include "./engine/comllvmrt/WrappedLLVMRuntime.h"

IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifvmCreateLLVMRuntimeBuilder()
    IFRIT_EXPORT_COMPAT_NOTHROW {
  return new Ifrit::Engine::ComLLVMRuntime::WrappedLLVMRuntimeBuilder();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifvmDestroyLLVMRuntimeBuilder(void *p)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  delete (Ifrit::Engine::ComLLVMRuntime::WrappedLLVMRuntimeBuilder *)p;
}