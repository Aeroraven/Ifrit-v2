#pragma once
#include <memory>
#include "./presentation/backend/BackendProvider.h"
namespace Ifrit::Presentation::Backend {
class IFRIT_APIDECL AdaptiveBackendBuilder {
public:
  std::unique_ptr<BackendProvider> buildUniqueBackend();
};
} // namespace Ifrit::Presentation::Backend