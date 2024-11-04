#pragma once
#include "ifrit/rhi/common/RhiLayer.h"
namespace Ifrit::Core {
class IApplication {
public:
  virtual void onStart() = 0;
  virtual void onUpdate() = 0;
  virtual void onEnd() = 0;

  virtual Ifrit::GraphicsBackend::Rhi::RhiBackend *getRhiLayer() = 0;
};
} // namespace Ifrit::Core