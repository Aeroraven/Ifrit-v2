
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

namespace Ifrit
{
    struct Task
    {
        Fn<void(Task*, void*)> m_Execute;
        Task*                  m_Parent          = nullptr;
        Atomic<i32>            m_IncompleteCount = 0;
    };

    struct TaskWorkerAttributes;
    class TaskWorker : public NonCopyable
    {
    private:
        TaskWorkerAttributes* m_Attributes = nullptr;
    };

    struct TaskSchedulerAttributes;
    class TaskScheduler : public NonCopyable
    {
    private:
        TaskSchedulerAttributes* m_Attributes = nullptr;

    public:
        using TaskRef = RObjectPool<Task>::RObjectRef;
        TaskRef CreateTask(Fn<void(Task*, void*)> fn, Task* parent = nullptr);
    };

} // namespace Ifrit