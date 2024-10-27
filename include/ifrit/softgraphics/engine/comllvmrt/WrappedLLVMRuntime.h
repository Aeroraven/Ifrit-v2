#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/ShaderRuntime.h"

namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ComLLVMRuntime {
struct WrappedLLVMRuntimeContext;
class WrappedLLVMRuntime : public ShaderRuntime {
public:
  WrappedLLVMRuntime();
  ~WrappedLLVMRuntime();
  static void initLlvmBackend();
  virtual void loadIR(std::string irCode, std::string irIdentifier);
  virtual void *lookupSymbol(std::string symbol);
  virtual std::unique_ptr<ShaderRuntime> getThreadLocalCopy();

private:
  WrappedLLVMRuntimeContext *session;
};

class WrappedLLVMRuntimeBuilder : public ShaderRuntimeBuilder {
public:
  WrappedLLVMRuntimeBuilder();
  virtual std::unique_ptr<ShaderRuntime> buildRuntime() const override;
};
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ComLLVMRuntime