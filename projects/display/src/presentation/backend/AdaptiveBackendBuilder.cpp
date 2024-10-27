#include "ifrit/display/presentation/backend/AdaptiveBackendBuilder.h"
#include "ifrit/display/presentation/backend/OpenGLBackend.h"

namespace Ifrit::Presentation::Backend {
IFRIT_APIDECL std::unique_ptr<BackendProvider>
AdaptiveBackendBuilder::buildUniqueBackend() {
  auto obj = std::make_unique<OpenGLBackend>();
  return obj;
}
} // namespace Ifrit::Presentation::Backend