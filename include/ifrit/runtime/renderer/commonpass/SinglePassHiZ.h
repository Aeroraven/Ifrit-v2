#pragma once
#include "ifrit/runtime/common/Pch.h"

#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/scene/FrameCollector.h"

namespace Ifrit::Runtime
{

    class IFRIT_APIDECL SinglePassHiZPass
    {
        using ComputePass  = RHI::RhiComputePass;
        using GPUCmdBuffer = RHI::RhiCommandList;
        using GPUTexture   = RHI::RhiTexture;
        using GPUSampler   = RHI::RhiSampler;

    protected:
        ComputePass*  m_SinglePassHiZPassMin = nullptr;
        ComputePass*  m_SinglePassHiZPassMax = nullptr;
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