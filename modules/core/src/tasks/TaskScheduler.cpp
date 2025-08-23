
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
#include "ifrit/core/hal/HalHostConcurrency.h"
#include "ifrit/core/typing/EnumReflection.h"

namespace Ifrit::Task
{

    static HashMap<ENamedTaskThread, std::thread::id> sNamedWorkerToThreadIdMap;

    // Task
    IFRIT_APIDECL void                                Task::Execute()
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
    struct TaskWorkerAttributes
    {
        using TaskRef                               = TObjectPool<Task>::TObjectRef;
        TaskScheduler*                  m_Scheduler = nullptr;
        u32                             m_ThreadId  = 0;
        Atomic<ETaskWorkerState>        m_State     = ETaskWorkerState::Alive;
        TPooledConcurrentQueue<TaskRef> m_JobQueue;
    };

    IFRIT_APIDECL TaskWorker::TaskWorker(TaskScheduler* scheduler, u32 threadId)
    {
        m_Attributes              = new TaskWorkerAttributes();
        m_Attributes->m_Scheduler = scheduler;
        m_Attributes->m_ThreadId  = threadId;
        m_Attributes->m_State     = ETaskWorkerState::Alive;
    }
    IFRIT_APIDECL      TaskWorker::~TaskWorker() { delete m_Attributes; }

    IFRIT_APIDECL void TaskWorker::Run()
    {
        while (true)
        {
            auto state = m_Attributes->m_State.load();
            if (state == ETaskWorkerState::Terminating)
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
        m_Attributes->m_State = ETaskWorkerState::Terminated;
    }

    IFRIT_APIDECL void TaskWorker::Launch()
    {
        m_Thread = std::thread([this]() {
            HAL::SetCurrentThreadId(m_Attributes->m_ThreadId + 1);
            if (mThreadType != ENamedTaskThread::AnyThread)
            {
                sNamedWorkerToThreadIdMap[mThreadType] = std::this_thread::get_id();
                IF_LOG_INFO("TaskScheduler", "Worker thread {} launched with thread type: {}", m_Attributes->m_ThreadId,
                    GetEnumName(mThreadType));
            }
            Run();
        });
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
        using TaskRef = TObjectPool<Task>::TObjectRef;
        // Hold this to ensure the object's reference count is not 0
        HashMap<FIndexedPtr::Underlying, TaskRef>    m_JobAlive;
        Vec<Owner<TaskWorker>>                       m_Workers;
        HashMap<ENamedTaskThread, Owner<TaskWorker>> m_NamedWorkers;
        TObjectPool<Task>                            m_TaskPool;
    };

    IFRIT_APIDECL TaskScheduler::TaskScheduler(u32 numThreads, bool isSingleton)
        : m_Attributes(new TaskSchedulerAttributes()), m_IsSingleton(isSingleton)
    {
        m_Attributes->m_Workers.reserve(numThreads);
        for (u32 i = 0; i < numThreads; ++i)
        {
            auto workerRef = MakeOwner<TaskWorker>(this, i);
            m_Attributes->m_Workers.push_back(std::move(workerRef));
            m_Attributes->m_Workers[i]->Launch();
        }
        IF_LOG_INFO("TaskScheduler", "Created {} worker threads.", numThreads);
    }

    IFRIT_APIDECL void TaskScheduler::DereferenceTask(FIndexedPtr taskId)
    {
        auto task = m_Attributes->m_JobAlive[taskId.Ptr()];
        if (task.Get() == nullptr)
        {
            IF_LOG_CRITICAL("TaskScheduler", "Task not found in alive task list. Task ID: {}", taskId.Ptr());
        }
        m_Attributes->m_JobAlive.erase(taskId.Ptr());
    }

    IFRIT_APIDECL bool TaskScheduler::RegisterDependency(Task* parent, Task* child)
    {
        // We do not need lock itself. The dependency is created upon creating.
        // No need to fear the deadlock
        FSpinLockGuard lockParent(parent->m_ContinuationLock);
        if (parent->m_State.load() == ETaskState::Idle)
        {
            // IF_LOG_CRITICAL("TaskScheduler", "To prevent circular dependency, the task is not allowed to be idle.");
        }
        if (parent->m_State.load() != ETaskState::Completed || parent->m_State.load() != ETaskState::Failed)
        {
            auto parentContPos = parent->m_ChildJobs.fetch_add(1);
            // parent->m_PendingJobs.fetch_add(1);
            parent->m_Continuations[parentContPos] = child;

            // For child
            auto childParPos                    = child->m_ParentJobs.fetch_add(1);
            child->m_Continuations[childParPos] = parent;
            return true;
        }
        return false;
    }

    IFRIT_APIDECL void TaskScheduler::ScheduleTask(TaskRef task)
    {
        auto        taskId       = task.GetIndex();
        auto        randomWorker = rand() % m_Attributes->m_Workers.size();

        TaskWorker* worker;
        auto        desiredThread = task->m_ThreadType;
        if (desiredThread == ENamedTaskThread::AnyThread)
        {
            worker = m_Attributes->m_Workers[randomWorker].get();
        }
        else
        {
            worker = GetNamedWorker(desiredThread);
        }

        if (worker->m_Attributes->m_State.load() == ETaskWorkerState::Alive)
        {
            worker->EnqueueTask(task);
        }
        else
        {
            IF_LOG_CRITICAL("TaskScheduler", "Worker is not alive. Task ID: {} will not be scheduled.", taskId.Ptr());
        }
    }

    IFRIT_APIDECL void TaskScheduler::ScheduleTaskFromId(FIndexedPtr taskId)
    {
        auto task = m_Attributes->m_JobAlive[taskId.Ptr()];
        if (task.Get() == nullptr)
        {
            IF_LOG_CRITICAL("TaskScheduler", "Task not found in alive task list. Task ID: {}", taskId.Ptr());
        }
        ScheduleTask(task);
    }

    IFRIT_APIDECL TaskWorker* TaskScheduler::FetchRandomWorker()
    {
        auto randomWorker = rand() % m_Attributes->m_Workers.size();
        return m_Attributes->m_Workers[randomWorker].get();
    }
    IFRIT_APIDECL TaskWorker* TaskScheduler::GetNamedWorker(ENamedTaskThread threadType)
    {
        IF_LOG_ASSERTION("TaskScheduler", threadType != ENamedTaskThread::Invalid,
            "Task thread type is invalid. Thread type: {}", static_cast<u32>(threadType));
        IF_LOG_ASSERTION("TaskScheduler", threadType != ENamedTaskThread::AnyThread,
            "Task thread type cannot be AnyThread. Thread type: {}", static_cast<u32>(threadType));
        IF_LOG_ASSERTION("TaskScheduler", m_Attributes->m_NamedWorkers.contains(threadType),
            "Task thread type is not registered. Thread type: {}", static_cast<u32>(threadType));
        auto worker = m_Attributes->m_NamedWorkers[threadType].get();
        if (worker->m_Attributes->m_State.load() == ETaskWorkerState::Alive)
        {
            return worker;
        }
        else
        {
            IF_LOG_CRITICAL("TaskScheduler", "Worker is not alive. Thread type: {}", static_cast<u32>(threadType));
            return nullptr;
        }
    }

    IFRIT_APIDECL TaskScheduler::TaskRef TaskScheduler::EnqueueTask(
        Fn<void(Task*, void*)> fn, ENamedTaskThread threadType, Vec<TaskRef> dependencies, void* payload)
    {
        auto task                              = m_Attributes->m_TaskPool.Create();
        auto taskId                            = task.GetIndex();
        task->m_Scheduler                      = this;
        task->m_PooledIdx                      = taskId;
        m_Attributes->m_JobAlive[taskId.Ptr()] = task;
        task->m_Execute                        = fn;
        task->m_Payload                        = payload;
        task->m_ThreadType                     = threadType;

        IF_LOG_ASSERTION("TaskScheduler", threadType != ENamedTaskThread::Invalid,
            "Task thread type is invalid. Task ID: {}", taskId.Ptr());

        bool hasDependency = false;
        for (auto& dep : dependencies)
        {
            hasDependency = RegisterDependency(dep.Get(), task.Get());
        }
        if (!hasDependency)
            ScheduleTask(task);
        return task;
    }

    IFRIT_APIDECL void TaskScheduler::WaitForTask(TaskRef task)
    {
        while (task->m_State.load() != ETaskState::Completed && task->m_State.load() != ETaskState::Failed)
        {
            std::this_thread::yield();
        }
    }

    IFRIT_APIDECL void TaskScheduler::RegisterNamedWorker(Owner<TaskWorker> worker, ENamedTaskThread threadType)
    {

        IF_LOG_ASSERTION("TaskScheduler", threadType != ENamedTaskThread::Invalid,
            "Task thread type is invalid. Thread type: {}", static_cast<u32>(threadType));
        IF_LOG_ASSERTION("TaskScheduler", threadType != ENamedTaskThread::AnyThread,
            "Task thread type cannot be AnyThread. Thread type: {}", static_cast<u32>(threadType));
        IF_LOG_ASSERTION("TaskScheduler", !m_Attributes->m_NamedWorkers.contains(threadType),
            "Task thread type is already registered. Thread type: {}", GetEnumName(threadType));
        IF_LOG_ASSERTION("TaskScheduler", worker->mThreadType == threadType,
            "Worker thread type does not match the registered thread type. Worker thread type: {}, "
            "Registered thread type: {}",
            GetEnumName(worker->mThreadType), GetEnumName(threadType));
        m_Attributes->m_NamedWorkers[threadType] = std::move(worker);
        m_Attributes->m_NamedWorkers[threadType]->Launch();
    }

    IFRIT_APIDECL TaskScheduler::~TaskScheduler()
    {
        if (m_IsSingleton)
        {
            return;
        }
        for (auto& worker : m_Attributes->m_Workers)
        {
            worker->m_Attributes->m_State = ETaskWorkerState::Terminating;
        }
        IF_LOG_INFO("TaskScheduler", "Waiting for all workers to finish...");
        for (auto& worker : m_Attributes->m_Workers)
        {
            while (worker->m_Attributes->m_State.load() != ETaskWorkerState::Terminated)
            {
                std::this_thread::yield();
            }
        }
        IF_LOG_INFO("TaskScheduler", "All workers finished.");
    }

    IFRIT_APIDECL TaskScheduler* GetTaskScheduler()
    {
        static TaskScheduler scheduler(8, true);
        return &scheduler;
    }

    IFRIT_APIDECL bool IsInNamedThread(ENamedTaskThread threadType)
    {
        IF_LOG_ASSERTION("TaskScheduler", threadType != ENamedTaskThread::Invalid,
            "Task thread type is invalid. Thread type: {}", static_cast<u32>(threadType));
        IF_LOG_ASSERTION("TaskScheduler", threadType != ENamedTaskThread::AnyThread,
            "Task thread type cannot be AnyThread. Thread type: {}", static_cast<u32>(threadType));
        auto threadId = std::this_thread::get_id();
        if (sNamedWorkerToThreadIdMap.contains(threadType))
        {
            return sNamedWorkerToThreadIdMap[threadType] == threadId;
        }
        else
        {
            IF_LOG_CRITICAL(
                "TaskScheduler", "Thread type is not registered. Thread type: {}", static_cast<u32>(threadType));
            return false;
        }
    }

} // namespace Ifrit::Task