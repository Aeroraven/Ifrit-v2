#pragma once
#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime
{
    namespace RHI
    {
        class RhiCommandList;
    }

    struct RenderingThreadData;
    struct RHIThreadData;

    class RenderingThread : public Task::TaskWorker
    {
    public:
        RenderingThread(Task::TaskScheduler* scheduler, u32 id);
        ~RenderingThread();

    private:
        RenderingThreadData* mData = nullptr;
    };

    class RHIThread : public Task::TaskWorker
    {
    public:
        RHIThread(Task::TaskScheduler* scheduler, u32 id);
        ~RHIThread();

    private:
        RHIThreadData* mData = nullptr;
    };

    IFRIT_RUNTIME_API Owner<Task::TaskWorker> GetRenderingThreadWorkerOwned();
    IFRIT_RUNTIME_API Owner<Task::TaskWorker> GetRHIThreadWorkerOwned();

    IFRIT_RUNTIME_API bool                    IsInRenderingThread();
    IFRIT_RUNTIME_API bool                    IsInRHIThread();

    IFRIT_RUNTIME_API void                    RegisterRenderCoreThreading();

} // namespace Ifrit::Runtime