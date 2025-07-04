
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/core/algo/ConcurrentQueue.h"

namespace Ifrit
{
    // Task
    IFRIT_APIDECL void Task::Execute()
    {
        m_State = ETaskState::Running;
        m_Execute(this, m_Payload);
        Complete();
    }

    IFRIT_APIDECL void Task::Complete()
    {
        // Complete Self
        {
            FSpinLockGuard lock(m_ContinuationLock);
            auto           jobsRemain = m_PendingJobs.fetch_sub(1, std::memory_order_acq_rel) - 1;
            m_State                   = ETaskState::Completed;
            if (jobsRemain == 0)
            {
                auto jobsRemain = m_PendingJobs.load(std::memory_order_acquire);
                if (jobsRemain == 0)
                {
                    auto numChildJobs = m_ChildJobs.load(std::memory_order_acquire);
                    // Decreases the parent job's child count
                    for (u32 i = 0; i < static_cast<u32>(numChildJobs); ++i)
                    {
                        auto childJob = m_Continuations[i];
                        if (childJob != nullptr)
                        {
                            auto childPending = childJob->m_ParentJobs.fetch_sub(1, std::memory_order_acq_rel);
                            childPending--;
                            if (childPending == 0)
                            {
                                m_Scheduler->ScheduleTaskFromId(childJob->m_PooledIdx);
                            }
                        }
                    }
                }
            }
        }

        // Notify parents. Note that if current task is running, the parent is
        // in Complete state, meaning its jobsRemain will never increase
        // TODO
    }

    // Workers
    struct FTaskWorkerAttributes
    {
        using TaskRef                               = TObjectPool<Task>::TObjectRef;
        FTaskScheduler*                 m_Scheduler = nullptr;
        u32                             m_ThreadId  = 0;
        Atomic<EFTaskWorkerState>       m_State     = EFTaskWorkerState::Alive;
        TPooledConcurrentQueue<TaskRef> m_JobQueue;
    };

    IFRIT_APIDECL FTaskWorker::FTaskWorker(FTaskScheduler* scheduler, u32 threadId)
    {
        m_Attributes              = new FTaskWorkerAttributes();
        m_Attributes->m_Scheduler = scheduler;
        m_Attributes->m_ThreadId  = threadId;
        m_Attributes->m_State     = EFTaskWorkerState::Alive;
    }
    IFRIT_APIDECL      FTaskWorker::~FTaskWorker() { delete m_Attributes; }

    IFRIT_APIDECL void FTaskWorker::Run()
    {
        while (true)
        {
            auto state = m_Attributes->m_State.load();
            if (state == EFTaskWorkerState::Terminating)
            {
                break;
            }
            else
            {
                auto task = FetchTask();
                if (task.Get() != nullptr)
                {
                    task->Execute();
                }
            }
            std::this_thread::yield();
        }
        m_Attributes->m_State = EFTaskWorkerState::Terminated;
    }

    IFRIT_APIDECL void FTaskWorker::Launch()
    {
        m_Thread = std::thread([this]() { Run(); });
        m_Thread.detach();
    }

    IFRIT_APIDECL void FTaskWorker::EnqueueTask(TaskRef task)
    {
        // Enqueue!
        m_Attributes->m_JobQueue.Enqueue(task);
    }

    IFRIT_APIDECL FTaskWorker::TaskRef FTaskWorker::FetchTask()
    {
        auto thisQueueTask = m_Attributes->m_JobQueue.Dequeue();
        if (thisQueueTask.Get() == nullptr)
        {
            auto workerToSteal = m_Attributes->m_Scheduler->FetchRandomWorker();
            if (workerToSteal != nullptr)
            {
                auto stolenTask = workerToSteal->m_Attributes->m_JobQueue.Dequeue();
                if (stolenTask.Get() != nullptr)
                {
                    return stolenTask;
                }
            }
            return TaskRef();
        }
        else
        {
            return thisQueueTask;
        }
    }
    // Scheduler

    struct FTaskSchedulerAttributes
    {
        using TaskRef = TObjectPool<Task>::TObjectRef;
        // Hold this to ensure the object's reference count is not 0
        HashMap<FIndexedPtr::Underlying, TaskRef> m_JobAlive;
        Vec<Ref<FTaskWorker>>                     m_Workers;
        TObjectPool<Task>                         m_TaskPool;
    };

    IFRIT_APIDECL FTaskScheduler::FTaskScheduler(u32 numThreads, bool isSingleton)
        : m_Attributes(new FTaskSchedulerAttributes()), m_IsSingleton(isSingleton)
    {
        m_Attributes->m_Workers.reserve(numThreads);
        for (u32 i = 0; i < numThreads; ++i)
        {
            auto workerRef = MakeRef<FTaskWorker>(this, i);
            m_Attributes->m_Workers.emplace_back(workerRef);
            m_Attributes->m_Workers[i]->Launch();
        }
        iInfo("FTaskScheduler: Created {} worker threads.", numThreads);
    }

    IFRIT_APIDECL void FTaskScheduler::DereferenceTask(FIndexedPtr taskId)
    {
        auto task = m_Attributes->m_JobAlive[taskId.Ptr()];
        if (task.Get() == nullptr)
        {
            iError("FTaskScheduler: Task not found in alive task list.");
            std::abort();
        }
        m_Attributes->m_JobAlive.erase(taskId.Ptr());
    }

    IFRIT_APIDECL void FTaskScheduler::RegisterDependency(Task* parent, Task* child)
    {
        // We do not need lock itself. The dependency is created upon creating.
        // No need to fear the deadlock
        FSpinLockGuard lockParent(parent->m_ContinuationLock);
        if (parent->m_State.load() == ETaskState::Idle)
        {
            iError("FTaskScheduler: To prevent circular dependency, the task is not allowed to be idle.");
            std::abort();
        }
        if (parent->m_State.load() != ETaskState::Completed || parent->m_State.load() != ETaskState::Failed)
        {
            auto parentContPos = parent->m_ChildJobs.fetch_add(1);
            // parent->m_PendingJobs.fetch_add(1);
            parent->m_Continuations[parentContPos] = child;

            // For child
            auto childParPos                    = child->m_ParentJobs.fetch_add(1);
            child->m_Continuations[childParPos] = parent;
        }
    }

    IFRIT_APIDECL void FTaskScheduler::ScheduleTask(TaskRef task)
    {
        auto taskId       = task.GetIndex();
        auto randomWorker = rand() % m_Attributes->m_Workers.size();
        auto worker       = m_Attributes->m_Workers[randomWorker];

        if (worker->m_Attributes->m_State.load() == EFTaskWorkerState::Alive)
        {
            worker->EnqueueTask(task);
        }
        else
        {
            iError("FTaskScheduler: Worker is not alive.");
            std::abort();
        }
    }

    IFRIT_APIDECL void FTaskScheduler::ScheduleTaskFromId(FIndexedPtr taskId)
    {
        auto task = m_Attributes->m_JobAlive[taskId.Ptr()];
        if (task.Get() == nullptr)
        {
            iError("FTaskScheduler: Task not found in alive task list.");
            std::abort();
        }
        ScheduleTask(task);
    }

    IFRIT_APIDECL FTaskWorker* FTaskScheduler::FetchRandomWorker()
    {
        auto randomWorker = rand() % m_Attributes->m_Workers.size();
        return m_Attributes->m_Workers[randomWorker].get();
    }

    IFRIT_APIDECL FTaskScheduler::TaskRef FTaskScheduler::EnqueueTask(
        Fn<void(Task*, void*)> fn, Vec<TaskRef> dependencies, void* payload)
    {
        auto task         = m_Attributes->m_TaskPool.Create();
        auto taskId       = task.GetIndex();
        task->m_Scheduler = this;
        task->m_PooledIdx = taskId;
        for (auto& dep : dependencies)
        {
            RegisterDependency(task.Get(), dep.Get());
        }
        m_Attributes->m_JobAlive[taskId.Ptr()] = task;
        task->m_Execute                        = fn;
        task->m_Payload                        = payload;

        // enqueue the task to a random worker
        ScheduleTask(task);
        return task;
    }

    IFRIT_APIDECL void FTaskScheduler::WaitForTask(TaskRef task)
    {
        while (task->m_State.load() != ETaskState::Completed && task->m_State.load() != ETaskState::Failed)
        {
            std::this_thread::yield();
        }
    }

    IFRIT_APIDECL FTaskScheduler::~FTaskScheduler()
    {
        if (m_IsSingleton)
        {
            return;
        }
        for (auto& worker : m_Attributes->m_Workers)
        {
            worker->m_Attributes->m_State = EFTaskWorkerState::Terminating;
        }
        iInfo("FTaskScheduler: Waiting for all workers to finish...");
        for (auto& worker : m_Attributes->m_Workers)
        {
            while (worker->m_Attributes->m_State.load() != EFTaskWorkerState::Terminated)
            {
                std::this_thread::yield();
            }
        }
        iInfo("FTaskScheduler: All workers finished.");
    }

    IFRIT_APIDECL FTaskScheduler* GetFTaskScheduler()
    {
        static FTaskScheduler scheduler(8, true);
        return &scheduler;
    }

} // namespace Ifrit