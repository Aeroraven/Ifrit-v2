#include "ifrit/vkrhi2/adapter/CommandSubmission.h"
#include "ifrit/vkrhi2/adapter/Device.h"
namespace Ifrit::RHI::VulkanRHI2
{
    struct VA_QueueSubmissionThreadInternal
    {
        VA_Device* mDevice = nullptr;
    };

    VA_QueueSubmissionThread::VA_QueueSubmissionThread(Task::TaskScheduler* scheduler, VA_Device* device, u32 id)
        : Task::TaskWorker(scheduler, id)
    {
        mData          = new VA_QueueSubmissionThreadInternal();
        mWorkerType    = Task::ETaskWorkerType::UniqueTask;
        mThreadType    = Task::ENamedTaskThread::RHISubmissionThread;
        mData->mDevice = device;
    }
    VA_QueueSubmissionThread::~VA_QueueSubmissionThread()
    {
        delete mData;
        mData = nullptr;
    }
    void VA_QueueSubmissionThread::RunUnique()
    {
        while (!IsTerminating())
        {
            auto activeQueue = mData->mDevice->GetActiveQueues();
            if (activeQueue.mGraphics)
                activeQueue.mGraphics->ProcessQueuedTasks();
            if (activeQueue.mAsyncCompute)
                activeQueue.mAsyncCompute->ProcessQueuedTasks();
            if (activeQueue.mTransfer)
                activeQueue.mTransfer->ProcessQueuedTasks();
        }
    }
} // namespace Ifrit::RHI::VulkanRHI2