#pragma once
#include "./SpvVMContext.h"
#include "./core/definition/CoreExports.h"
namespace Ifrit::Engine::SoftRenderer::ShaderVM::Spirv {
class SpvVMInterpreter {
public:
  void parseRawContext(SpvVMContext *context,
                       SpvVMIntermediateRepresentation *outIr);
  void exportLlvmIR(SpvVMIntermediateRepresentation *ir,
                    std::string *outLlvmIR);
};
} // namespace Ifrit::Engine::SoftRenderer::ShaderVM::Spirv