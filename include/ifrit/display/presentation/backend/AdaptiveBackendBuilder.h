#pragma once
#include "ifrit/display/presentation/backend/BackendProvider.h"
#include <memory>

namespace Ifrit::Display::Backend {
class IFRIT_APIDECL AdaptiveBackendBuilder {
public:
  std::unique_ptr<BackendProvider> buildUniqueBackend();
};
} // namespace Ifrit::Display::Backend