#include "ifrit/runtime/rendercore/threading/RenderingThread.h"
#include "ifrit/rhi/common/RhiLayer.h"

namespace Ifrit::Runtime
{

    struct RenderingThreadData
    {
    };

    struct RHIThreadData
    {
    };

    RenderingThread::RenderingThread(Task::TaskScheduler* scheduler, u32 id) : Task::TaskWorker(scheduler, id)
    {
        mData       = new RenderingThreadData();
        mThreadType = Task::ENamedTaskThread::RenderThread;
    }
    RenderingThread::~RenderingThread() { delete mData; }
    RHIThread::RHIThread(Task::TaskScheduler* scheduler, u32 id) : Task::TaskWorker(scheduler, id)
    {
        mData       = new RHIThreadData();
        mThreadType = Task::ENamedTaskThread::RHIThread;
    }
    RHIThread::~RHIThread() { delete mData; }

    IFRIT_RUNTIME_API Owner<Task::TaskWorker> GetRenderingThreadWorkerOwned()
    {
        return MakeOwner<RenderingThread>(Ifrit::Task::GetTaskScheduler(), 1919810);
    }
    IFRIT_RUNTIME_API Owner<Task::TaskWorker> GetRHIThreadWorkerOwned()
    {
        return MakeOwner<RHIThread>(Ifrit::Task::GetTaskScheduler(), 114514);
    }

    IFRIT_RUNTIME_API void RegisterRenderCoreThreading()
    {
        Ifrit::Task::GetTaskScheduler()->RegisterNamedWorker(
            GetRenderingThreadWorkerOwned(), Task::ENamedTaskThread::RenderThread);
        Ifrit::Task::GetTaskScheduler()->RegisterNamedWorker(
            GetRHIThreadWorkerOwned(), Task::ENamedTaskThread::RHIThread);
    }

    IFRIT_RUNTIME_API bool IsInRenderingThread()
    {
        return Ifrit::Task::IsInNamedThread(Task::ENamedTaskThread::RenderThread);
    }
    IFRIT_RUNTIME_API bool IsInRHIThread() { return Ifrit::Task::IsInNamedThread(Task::ENamedTaskThread::RHIThread); }

} // namespace Ifrit::Runtime