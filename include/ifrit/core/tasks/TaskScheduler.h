
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
#include "ifrit/core/base/containers/Atomic.h"

namespace Ifrit::Task
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

    enum class ETaskWorkerState : u32
    {
        Alive,
        Terminating,
        Terminated,
    };

    enum class ENamedTaskThread : u32
    {
        Invalid             = 0,
        GameThread          = 1,
        RenderThread        = 2,
        RHIThread           = 3,
        RHISubmissionThread = 4,
        AnyThread           = 0xff
    };

    enum class ETaskWorkerType : u32
    {
        Generic    = 0,
        UniqueTask = 1
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

        TAtomic<i32>                            m_PendingJobs = 1;
        TAtomic<i32>                            m_ChildJobs   = 0;
        TAtomic<i32>                            m_ParentJobs  = 0;

        FSpinLock                               m_ContinuationLock = 0;
        TAtomic<ETaskState>                     m_State            = ETaskState::Idle;
        Array<Task*, cTaskMaxContinuationCount> m_Continuations;
        Array<Task*, cTaskMaxContinuationCount> m_Parents;
        FIndexedPtr                             m_PooledIdx = FIndexedPtr(0);
        TaskScheduler*                          m_Scheduler = nullptr;

        ENamedTaskThread                        m_ThreadType = ENamedTaskThread::Invalid;

        void*                                   m_Payload;

    public:
        void          Execute();
        void          Complete();
        void          Finalize();

        inline IntPtr GetId() { return m_PooledIdx.Ptr(); }

        friend TaskScheduler;
    };

    using TaskReference = TObjectPool<Task>::TObjectRef;

    struct TaskWorkerAttributes;
    class IFRIT_APIDECL TaskWorker : public NonCopyable
    {
        using TaskRef = TObjectPool<Task>::TObjectRef;

    private:
        void    EnqueueTask(TaskRef task);
        TaskRef FetchTask();

    public:
        TaskWorker(TaskScheduler* scheduler, u32 id);
        ~TaskWorker();

        void         Launch();
        void         Run();
        void         RequestTerminate();

        virtual void RunUnique() {}

        bool         IsTerminating() const;
        void         WaitForTermination();

        friend class TaskScheduler;

    protected:
        ENamedTaskThread mThreadType = ENamedTaskThread::AnyThread;
        ETaskWorkerType  mWorkerType = ETaskWorkerType::Generic;

    private:
        std::thread           m_Thread;
        TaskWorkerAttributes* m_Attributes = nullptr;
    };
    using TaskHandle = TObjectPool<Task>::TObjectRef;

    struct TaskSchedulerAttributes;
    class IFRIT_APIDECL TaskScheduler : public NonCopyable
    {
    private:
        // I don't want the use of dangled pointer
        using TaskRef = TObjectPool<Task>::TObjectRef;

    private:
        bool        RegisterDependency(Task* parent, Task* child);
        void        ScheduleTask(TaskRef task);
        void        ScheduleTaskFromId(FIndexedPtr taskId);
        TaskWorker* FetchRandomWorker();
        void        DereferenceTask(FIndexedPtr taskId);
        TaskWorker* GetNamedWorker(ENamedTaskThread threadType);

    public:
        TaskScheduler(u32 numWorkers, bool isSingleton = false);
        ~TaskScheduler();
        TaskRef EnqueueTask(
            Fn<void(Task*, void*)> fn, ENamedTaskThread threadType, Vec<TaskRef> dependencies, void* payload);
        void WaitForTask(TaskRef task);
        void RegisterNamedWorker(Owner<TaskWorker> worker, ENamedTaskThread threadType);
        void RequestTerminating(ENamedTaskThread threadType);
        void WaitForTerminating(ENamedTaskThread threadType);

        friend class TaskWorker;
        friend class Task;

    private:
        TaskSchedulerAttributes* m_Attributes  = nullptr;
        bool                     m_IsSingleton = false;
    };

    IFRIT_CORE_API TaskScheduler* GetTaskScheduler();
    IFRIT_CORE_API bool           IsInNamedThread(ENamedTaskThread threadType);

} // namespace Ifrit::Task