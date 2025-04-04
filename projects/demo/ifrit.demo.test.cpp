#include "ifrit/core/algo/Memory.h"
#include "ifrit/core/algo/ConcurrentQueue.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/rhi/platform/RhiSelector.h"
using namespace Ifrit;
using namespace Ifrit::Graphics::Rhi;

namespace Ifrit::Test
{
    class Cat
    {
    private:
        int    m_id;
        String m_name;

    public:
        Cat(int id, String name) : m_id(id), m_name(name) {}
        void Print() const { iInfo("Cat id: {}, name: {}", m_id, m_name); }
        int  GetId() const { return m_id; }
    };
} // namespace Ifrit::Test

void RpoolTest()
{
    RPooledConcurrentQueue<Ifrit::Test::Cat> q;
    // Start 2 threads,each deque 500 elements
    Atomic<u64>                              count = 0;
    Ifrit::UnorderedFor<int>(0, 16, [&](int i) {
        for (int i = 0; i < 100000; i++)
        {
            q.Enqueue(Ifrit::Test::Cat(i, "Cat" + std::to_string(i)));
        }
        while (!q.Empty())
        {
            auto cat = q.Dequeue();
            printf("Thread %d, cat id: %d\n", i, cat.GetId());
            count.fetch_add(1, std::memory_order::acq_rel);
        }
    });
    printf("Total dequeued: %llu\n", count.load(std::memory_order::acquire));
}

void taskTest()
{
    TaskScheduler   scheduler(8);
    Vec<TaskHandle> tasks;
    iInfo("Starting task test");
    for (int i = 0; i < 100; i++)
    {
        auto task = scheduler.EnqueueTask(
            [&scheduler, &tasks, i](Task* task, void*) {
                printf("Task %d is running\n", task->GetId());
                std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000));
            },
            {}, nullptr);
        tasks.push_back(task);
    }

    // Wait for all tasks to complete
    for (auto& task : tasks)
    {
        scheduler.WaitForTask(task);
    }
}

int main()
{
    taskTest();
    return 0;
}