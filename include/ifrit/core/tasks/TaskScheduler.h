
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

#include "ifrit/core/base/CoreBase.h"
namespace Ifrit
{
    IF_CONSTEXPR u32 cTaskMaxContinuationCount = 16;

    enum class ETaskState : u32
    {
        Idle,
        Scheduling,
        Running,
        Completed,
        Failed,
    };

    enum class EFTaskWorkerState : u32
    {
        Alive,
        Terminating,
        Terminated,
    };

    class FTaskScheduler;

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

        FSpinLock                               m_ContinuationLock = 0;
        Atomic<ETaskState>                      m_State            = ETaskState::Idle;
        Array<Task*, cTaskMaxContinuationCount> m_Continuations;
        Array<Task*, cTaskMaxContinuationCount> m_Parents;
        FIndexedPtr                             m_PooledIdx = FIndexedPtr(0);
        FTaskScheduler*                         m_Scheduler = nullptr;

        void*                                   m_Payload;

    public:
        void          Execute();
        void          Complete();
        void          Finalize();

        inline IntPtr GetId() { return m_PooledIdx.Ptr(); }

        friend FTaskScheduler;
    };

    struct FTaskWorkerAttributes;
    class IFRIT_APIDECL FTaskWorker : public NonCopyable
    {
        using TaskRef = TObjectPool<Task>::TObjectRef;

    private:
        std::thread            m_Thread;
        FTaskWorkerAttributes* m_Attributes = nullptr;

    private:
        void    EnqueueTask(TaskRef task);
        TaskRef FetchTask();

    public:
        FTaskWorker(FTaskScheduler* scheduler, u32 id);
        ~FTaskWorker();

        void Launch();
        void Run();

        friend class FTaskScheduler;
    };
    using TaskHandle = TObjectPool<Task>::TObjectRef;

    struct FTaskSchedulerAttributes;
    class IFRIT_APIDECL FTaskScheduler : public NonCopyable
    {
    private:
        // I don't want the use of dangled pointer
        using TaskRef = TObjectPool<Task>::TObjectRef;

    private:
        FTaskSchedulerAttributes* m_Attributes  = nullptr;
        bool                      m_IsSingleton = false;

    private:
        void         RegisterDependency(Task* parent, Task* child);
        void         ScheduleTask(TaskRef task);
        void         ScheduleTaskFromId(FIndexedPtr taskId);
        FTaskWorker* FetchRandomWorker();
        void         DereferenceTask(FIndexedPtr taskId);

    public:
        FTaskScheduler(u32 numWorkers, bool isSingleton = false);
        ~FTaskScheduler();
        TaskRef EnqueueTask(Fn<void(Task*, void*)> fn, Vec<TaskRef> dependencies, void* payload);
        void    WaitForTask(TaskRef task);

        friend class FTaskWorker;
        friend class Task;
    };

    IFRIT_CORE_API FTaskScheduler* GetFTaskScheduler();

} // namespace Ifrit