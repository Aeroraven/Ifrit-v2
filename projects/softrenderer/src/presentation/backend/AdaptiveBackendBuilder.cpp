#include "./presentation/backend/AdaptiveBackendBuilder.h"
#include "./presentation/backend/OpenGLBackend.h"

namespace Ifrit::Presentation::Backend {
std::unique_ptr<BackendProvider> AdaptiveBackendBuilder::buildUniqueBackend() {
  auto obj = std::make_unique<OpenGLBackend>();
  return obj;
}
} // namespace Ifrit::Presentation::Backend