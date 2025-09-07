#pragma once

#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/application/Subsystem.h"
#include "ifrit/profiler/ProfilerBase.h"
#include "ifrit/profiler/ProfilerConfig.h"
namespace Ifrit::Profiler
{

    struct ProfilerSystemData;
    class IFRIT_PROFILER_API ProfilerSystem : public Runtime::ISubsystem
    {
    private:
        ProfilerConfig      mConfig;
        ProfilerSystemData* mData;

    public:
        ProfilerSystem();
        ~ProfilerSystem() override;

        virtual void                          OnInitialize(Runtime::IApplication* app) override;
        virtual void                          OnShutdown() override;
        virtual void                          OnFrameBegin() override;
        virtual void                          OnFrameEnd() override;
        virtual Owner<RHI::RhiTaskSubmission> OnPreRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual Owner<RHI::RhiTaskSubmission> OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual void                          OnUpdate(Runtime::Scene* scene) override;

    public:
        inline static Owner<ProfilerSystem> Create(ProfilerConfig config = ProfilerConfig())
        {
            auto profilerSystem     = MakeOwner<ProfilerSystem>();
            profilerSystem->mConfig = config;
            return profilerSystem;
        }
    };
} // namespace Ifrit::Profiler