#pragma once
#include "core/data/Image.h"
#include "core/definition/CoreExports.h"
#include "presentation/backend/BackendProvider.h"

namespace Ifrit::Presentation::Backend {
class AbstractTerminalBackend : public BackendProvider {
public:
  virtual void setCursor(int x, int y, std::string &str);
};
} // namespace Ifrit::Presentation::Backend