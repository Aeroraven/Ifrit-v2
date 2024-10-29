#pragma once
#include "./SpvVMContext.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"
namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::Spirv {
class SpvVMInterpreter {
public:
  void parseRawContext(SpvVMContext *context,
                       SpvVMIntermediateRepresentation *outIr);
  void exportLlvmIR(SpvVMIntermediateRepresentation *ir,
                    std::string *outLlvmIR);
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics::ShaderVM::Spirv