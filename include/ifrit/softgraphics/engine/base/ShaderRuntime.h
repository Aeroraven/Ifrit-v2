#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
namespace Ifrit::GraphicsBackend::SoftGraphics {

class ShaderRuntime {
public:
  virtual ~ShaderRuntime() = default;
  virtual void loadIR(std::string shaderCode, std::string shaderIdentifier) = 0;
  virtual void *lookupSymbol(std::string symbol) = 0;
  virtual std::unique_ptr<ShaderRuntime> getThreadLocalCopy() = 0;
};

class ShaderRuntimeBuilder {
public:
  virtual std::unique_ptr<ShaderRuntime> buildRuntime() const = 0;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics