
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

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/algo/Memory.h"
#include "ifrit/core/algo/Parallel.h"

namespace Ifrit
{
    IF_CONSTEXPR u32 cTaskMaxContinuationCount = 16;

    enum class TaskState : u32
    {
        Idle,
        Scheduling,
        Running,
        Completed,
        Failed,
    };

    enum class TaskWorkerState : u32
    {
        Alive,
        Terminating,
        Terminated,
    };

    class TaskScheduler;

    class IFRIT_APIDECL Task
    {
    private:
        Fn<void(Task*, void*)>                  m_Execute;

        // Unity supports mulitiple dependencies,
        // I cannot figure out any lock-free plan to do this. Now, the rough plan is
        //
        // 1. When parent task finishes, directly enter a critical zone
        // 2. Set state flag, and enqueue all child tasks to the queue
        // 3. Release the lock
        //
        // For children, the parent will only have 2 states: running or completed.
        // 1. Acquire parent's lock, if parent is completed, discard this parent
        // 2. If parent is running, add to parent's child list
        // 3. Release the lock

        Atomic<i32>                             m_PendingJobs = 1;
        Atomic<i32>                             m_ChildJobs   = 0;
        Atomic<i32>                             m_ParentJobs  = 0;

        RSpinLock                               m_ContinuationLock = 0;
        Atomic<TaskState>                       m_State            = TaskState::Idle;
        Array<Task*, cTaskMaxContinuationCount> m_Continuations;
        Array<Task*, cTaskMaxContinuationCount> m_Parents;
        RIndexedPtr                             m_PooledIdx = RIndexedPtr(0);
        TaskScheduler*                          m_Scheduler = nullptr;

        void*                                   m_Payload;

    public:
        void          Execute();
        void          Complete();
        void          Finalize();

        inline IntPtr GetId() { return m_PooledIdx.Ptr(); }

        friend TaskScheduler;
    };

    struct TaskWorkerAttributes;
    class IFRIT_APIDECL TaskWorker : public NonCopyable
    {
        using TaskRef = RObjectPool<Task>::RObjectRef;

    private:
        std::thread           m_Thread;
        TaskWorkerAttributes* m_Attributes = nullptr;

    private:
        void    EnqueueTask(TaskRef task);
        TaskRef FetchTask();

    public:
        TaskWorker(TaskScheduler* scheduler, u32 id);
        ~TaskWorker();

        void Launch();
        void Run();

        friend class TaskScheduler;
    };
    using TaskHandle = RObjectPool<Task>::RObjectRef;

    struct TaskSchedulerAttributes;
    class IFRIT_APIDECL TaskScheduler : public NonCopyable
    {
    private:
        // I don't want the use of dangled pointer
        using TaskRef = RObjectPool<Task>::RObjectRef;

    private:
        TaskSchedulerAttributes* m_Attributes = nullptr;

    private:
        void        RegisterDependency(Task* parent, Task* child);
        void        ScheduleTask(TaskRef task);
        void        ScheduleTaskFromId(RIndexedPtr taskId);
        TaskWorker* FetchRandomWorker();
        void        DereferenceTask(RIndexedPtr taskId);

    public:
        TaskScheduler(u32 numWorkers);
        ~TaskScheduler();
        TaskRef EnqueueTask(Fn<void(Task*, void*)> fn, Vec<TaskRef> dependencies, void* payload);
        void    WaitForTask(TaskRef task);

        friend class TaskWorker;
        friend class Task;
    };

} // namespace Ifrit