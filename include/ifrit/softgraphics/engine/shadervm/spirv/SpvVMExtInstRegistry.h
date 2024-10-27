#pragma once
#include "ifrit/softgraphics/core/definition/CoreExports.h"
namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ShaderVM::Spirv {
enum SpvVMExtRegistryTypeIdentifier {
  IFSP_EXTREG_TP_INT,
  IFSP_EXTREG_TP_FLOAT
};

typedef std::string (*SpvVMExtRegistryFunctionGenerator)(
    const std::vector<SpvVMExtRegistryTypeIdentifier> &,
    const std::vector<int> &);

class SpvVMExtRegistry {
private:
  std::unordered_map<std::string,
                     std::unordered_map<int, SpvVMExtRegistryFunctionGenerator>>
      generators;
  std::unordered_set<std::string> registeredFunc;
  std::string irCode;

public:
  SpvVMExtRegistry();
  std::string queryExternalFunc(
      std::string extImportName, int functionName,
      const std::vector<SpvVMExtRegistryTypeIdentifier> &identifiers,
      const std::vector<int> &componentSize);
  std::string getRequiredFuncDefs();
};
} // namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::ShaderVM::Spirv