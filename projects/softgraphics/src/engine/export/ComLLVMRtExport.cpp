#include "ifrit/softgraphics/engine/export/ComLLVMRtExport.h"
#include "ifrit/softgraphics/engine/comllvmrt/WrappedLLVMRuntime.h"

IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifvmCreateLLVMRuntimeBuilder()
    IFRIT_EXPORT_COMPAT_NOTHROW {
  return new Ifrit::GraphicsBackend::SoftGraphics::ComLLVMRuntime::
      WrappedLLVMRuntimeBuilder();
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifvmDestroyLLVMRuntimeBuilder(void *p)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  delete (Ifrit::GraphicsBackend::SoftGraphics::ComLLVMRuntime::
              WrappedLLVMRuntimeBuilder *)p;
}