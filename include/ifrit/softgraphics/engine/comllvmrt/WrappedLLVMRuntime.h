
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


#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
#include "ifrit/softgraphics/engine/base/ShaderRuntime.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::ComLLVMRuntime {
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
} // namespace Ifrit::GraphicsBackend::SoftGraphics::ComLLVMRuntime