#pragma once
#include "ifrit/display/presentation/backend/BackendProvider.h"
#include <string>


namespace Ifrit::Presentation::Backend {
class IFRIT_APIDECL AbstractTerminalBackend : public BackendProvider {
public:
  virtual void setCursor(int x, int y, std::string &str);
};
} // namespace Ifrit::Presentation::Backend