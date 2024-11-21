#include "ifrit/core/scene/SceneManager.h"
#include "ifrit/common/util/TypingUtil.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::Core {
IFRIT_APIDECL void
SceneManager::collectPerframeData(PerFrameData &perframeData, Scene *scene,
                                  Camera *camera,
                                  GraphicsShaderPassType passType) {
  throw std::runtime_error("Deprecated");
}
} // namespace Ifrit::Core