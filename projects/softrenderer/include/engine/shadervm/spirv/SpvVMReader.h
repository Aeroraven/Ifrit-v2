#pragma once
#include "./SpvVMContext.h"
#include "./core/definition/CoreExports.h"
namespace Ifrit::Engine::ShaderVM::Spirv {
class SpvVMReader {
public:
  std::vector<char> readFile(const char *path);
  void initializeContext(SpvVMContext *outContext);
  void parseByteCode(const char *byteCode, size_t length,
                     SpvVMContext *outContext);
  void printParsedInstructions(SpvVMContext *outContext);
};
} // namespace Ifrit::Engine::ShaderVM::Spirv