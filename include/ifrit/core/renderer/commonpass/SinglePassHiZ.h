#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/core/base/ApplicationInterface.h"
#include "ifrit/core/base/Scene.h"
#include "ifrit/core/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Core {

class IFRIT_APIDECL SinglePassHiZPass {
  using ComputePass = Ifrit::GraphicsBackend::Rhi::RhiComputePass;
  using GPUCmdBuffer = Ifrit::GraphicsBackend::Rhi::RhiCommandBuffer;
  using GPUTexture = Ifrit::GraphicsBackend::Rhi::RhiTexture;
  using GPUSampler = Ifrit::GraphicsBackend::Rhi::RhiSampler;

protected:
  ComputePass *m_singlePassHiZPass = nullptr;
  IApplication *m_app;

public:
  SinglePassHiZPass(IApplication *app);
  virtual void prepareHiZResources(PerFrameData::SinglePassHiZData &data, GPUTexture *depthTexture, GPUSampler *sampler,
                                   u32 rtWidth, u32 rtHeight);
  virtual bool checkResourceToRebuild(PerFrameData::SinglePassHiZData &data, u32 rtWidth, u32 rtHeight);
  virtual void runHiZPass(const PerFrameData::SinglePassHiZData &data, const GPUCmdBuffer *cmd, u32 rtWidth,
                          u32 rtHeight, bool minMode);
};
} // namespace Ifrit::Core