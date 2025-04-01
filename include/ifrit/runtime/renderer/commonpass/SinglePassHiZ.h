#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/scene/FrameCollector.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Runtime
{

    class IFRIT_APIDECL SinglePassHiZPass
    {
        using ComputePass  = Graphics::Rhi::RhiComputePass;
        using GPUCmdBuffer = Graphics::Rhi::RhiCommandList;
        using GPUTexture   = Graphics::Rhi::RhiTexture;
        using GPUSampler   = Graphics::Rhi::RhiSampler;

    protected:
        ComputePass*  m_singlePassHiZPass = nullptr;
        IApplication* m_app;

    public:
        SinglePassHiZPass(IApplication* app);

        virtual void PrepareHiZResources(PerFrameData::SinglePassHiZData& data, GPUTexture* depthTexture,
            GPUSampler* sampler, u32 rtWidth, u32 rtHeight);
        virtual bool CheckResourceToRebuild(PerFrameData::SinglePassHiZData& data, u32 rtWidth, u32 rtHeight);
        virtual void RunHiZPass(const PerFrameData::SinglePassHiZData& data, const GPUCmdBuffer* cmd, u32 rtWidth,
            u32 rtHeight, bool minMode);
    };
} // namespace Ifrit::Runtime