
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
        m_State = TaskState::Running;
        m_Execute(this, m_Payload);
        Complete();
    }

    IFRIT_APIDECL void Task::Complete()
    {
        // Complete Self
        {
            RSpinLockGuard lock(m_ContinuationLock);
            auto           jobsRemain = m_PendingJobs.fetch_sub(1, std::memory_order_acq_rel) - 1;
            m_State                   = TaskState::Completed;
            if (jobsRemain == 0)
            {
                auto jobsRemain = m_PendingJobs.load(std::memory_order_acquire);
                if (jobsRemain == 0)
                {
                    auto numChildJobs = m_ChildJobs.load(std::memory_order_acquire);
                    // Decreases the parent job's child count
                    for (u32 i = 0; i < numChildJobs; ++i)
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
    struct TaskWorkerAttributes
    {
        using TaskRef                               = RObjectPool<Task>::RObjectRef;
        TaskScheduler*                  m_Scheduler = nullptr;
        u32                             m_ThreadId  = 0;
        Atomic<TaskWorkerState>         m_State     = TaskWorkerState::Alive;
        RPooledConcurrentQueue<TaskRef> m_JobQueue;
    };

    IFRIT_APIDECL TaskWorker::TaskWorker(TaskScheduler* scheduler, u32 threadId)
    {
        m_Attributes              = new TaskWorkerAttributes();
        m_Attributes->m_Scheduler = scheduler;
        m_Attributes->m_ThreadId  = threadId;
        m_Attributes->m_State     = TaskWorkerState::Alive;
    }
    IFRIT_APIDECL      TaskWorker::~TaskWorker() { delete m_Attributes; }

    IFRIT_APIDECL void TaskWorker::Run()
    {
        while (true)
        {
            auto state = m_Attributes->m_State.load();
            if (state == TaskWorkerState::Terminating)
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
        m_Attributes->m_State = TaskWorkerState::Terminated;
    }

    IFRIT_APIDECL void TaskWorker::Launch()
    {
        m_Thread = std::thread([this]() { Run(); });
        m_Thread.detach();
    }

    IFRIT_APIDECL void TaskWorker::EnqueueTask(TaskRef task)
    {
        // Enqueue!
        m_Attributes->m_JobQueue.Enqueue(task);
    }

    IFRIT_APIDECL TaskWorker::TaskRef TaskWorker::FetchTask()
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
            int sz = m_Attributes->m_JobQueue.Size();
            int th = m_Attributes->m_ThreadId;
            // iWarn("TaskWorker: No task to fetch {}. Worker {} is idle.",sz ,th );
            return TaskRef();
        }
        else
        {
            return thisQueueTask;
        }
    }
    // Scheduler

    struct TaskSchedulerAttributes
    {
        using TaskRef = RObjectPool<Task>::RObjectRef;
        // Hold this to ensure the object's reference count is not 0
        HashMap<RIndexedPtr::Underlying, TaskRef> m_JobAlive;
        Vec<Ref<TaskWorker>>                      m_Workers;
        RObjectPool<Task>                         m_TaskPool;
    };

    IFRIT_APIDECL TaskScheduler::TaskScheduler(u32 numThreads) : m_Attributes(new TaskSchedulerAttributes())
    {
        m_Attributes->m_Workers.reserve(numThreads);
        for (u32 i = 0; i < numThreads; ++i)
        {
            auto workerRef = std::make_shared<TaskWorker>(this, i);
            m_Attributes->m_Workers.emplace_back(workerRef);
            m_Attributes->m_Workers[i]->Launch();
        }
    }

    IFRIT_APIDECL void TaskScheduler::DereferenceTask(RIndexedPtr taskId)
    {
        auto task = m_Attributes->m_JobAlive[taskId.Ptr()];
        if (task.Get() == nullptr)
        {
            iError("TaskScheduler: Task not found in alive task list.");
            std::abort();
        }
        m_Attributes->m_JobAlive.erase(taskId.Ptr());
    }

    IFRIT_APIDECL void TaskScheduler::RegisterDependency(Task* parent, Task* child)
    {
        // We do not need lock itself. The dependency is created upon creating.
        // No need to fear the deadlock
        RSpinLockGuard lockParent(parent->m_ContinuationLock);
        if (parent->m_State.load() == TaskState::Idle)
        {
            iError("TaskScheduler: To prevent circular dependency, the task is not allowed to be idle.");
            std::abort();
        }
        if (parent->m_State.load() != TaskState::Completed || parent->m_State.load() != TaskState::Failed)
        {
            auto parentContPos = parent->m_ChildJobs.fetch_add(1);
            // parent->m_PendingJobs.fetch_add(1);
            parent->m_Continuations[parentContPos] = child;

            // For child
            auto childParPos                    = child->m_ParentJobs.fetch_add(1);
            child->m_Continuations[childParPos] = parent;
        }
    }

    IFRIT_APIDECL void TaskScheduler::ScheduleTask(TaskRef task)
    {
        auto taskId       = task.GetIndex();
        auto randomWorker = rand() % m_Attributes->m_Workers.size();
        auto worker       = m_Attributes->m_Workers[randomWorker];

        if (worker->m_Attributes->m_State.load() == TaskWorkerState::Alive)
        {
            worker->EnqueueTask(task);
        }
        else
        {
            iError("TaskScheduler: Worker is not alive.");
            std::abort();
        }
    }

    IFRIT_APIDECL void TaskScheduler::ScheduleTaskFromId(RIndexedPtr taskId)
    {
        auto task = m_Attributes->m_JobAlive[taskId.Ptr()];
        if (task.Get() == nullptr)
        {
            iError("TaskScheduler: Task not found in alive task list.");
            std::abort();
        }
        ScheduleTask(task);
    }

    IFRIT_APIDECL TaskWorker* TaskScheduler::FetchRandomWorker()
    {
        auto randomWorker = rand() % m_Attributes->m_Workers.size();
        return m_Attributes->m_Workers[randomWorker].get();
    }

    IFRIT_APIDECL TaskScheduler::TaskRef TaskScheduler::EnqueueTask(
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

    IFRIT_APIDECL void TaskScheduler::WaitForTask(TaskRef task)
    {
        while (task->m_State.load() != TaskState::Completed && task->m_State.load() != TaskState::Failed)
        {
            std::this_thread::yield();
        }
    }

    IFRIT_APIDECL TaskScheduler::~TaskScheduler()
    {
        for (auto& worker : m_Attributes->m_Workers)
        {
            worker->m_Attributes->m_State = TaskWorkerState::Terminating;
        }
        iInfo("TaskScheduler: Waiting for all workers to finish...");
        for (auto& worker : m_Attributes->m_Workers)
        {
            while (worker->m_Attributes->m_State.load() != TaskWorkerState::Terminated)
            {
                std::this_thread::yield();
            }
        }
        iInfo("TaskScheduler: All workers finished.");
    }

} // namespace Ifrit