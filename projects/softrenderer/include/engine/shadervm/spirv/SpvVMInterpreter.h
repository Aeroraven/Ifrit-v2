#pragma once
#include "./SpvVMContext.h"
#include "./core/definition/CoreExports.h"
namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ShaderVM::Spirv {
class SpvVMInterpreter {
public:
  void parseRawContext(SpvVMContext *context,
                       SpvVMIntermediateRepresentation *outIr);
  void exportLlvmIR(SpvVMIntermediateRepresentation *ir,
                    std::string *outLlvmIR);
};
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ShaderVM::Spirv