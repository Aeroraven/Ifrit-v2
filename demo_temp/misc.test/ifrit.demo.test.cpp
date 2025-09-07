#include "ifrit/core/algo/Memory.h"
#include "ifrit/core/algo/ConcurrentQueue.h"
#include "ifrit/core/logging/Logging.h"
#include "ifrit/core/algo/Parallel.h"
#include "ifrit/core/tasks/TaskScheduler.h"
#include "ifrit/rhi/platform/RhiSelector.h"
#include "ifrit/geomproc/sampler/PoissonSampler.h"
#include "ifrit/geomproc/vdb/VdbBase.h"
#include "ifrit/geomproc/vdb/VdbSampler.h"
#include "ifrit/geomproc/pointcloud/PointCloudTransforms.h"
#include "ifrit/core/file/FileOps.h"
#include <iostream>
using namespace Ifrit;
using namespace Ifrit::RHI;

namespace Ifrit::Test
{
    class Cat
    {
    private:
        int    m_id;
        String m_name;

    public:
        Cat(int id, String name) : m_id(id), m_name(name) {}
        void Print() const { IF_LOG_INFO("DemoTest","Cat id: {}, name: {}", m_id, m_name); }
        int  GetId() const { return m_id; }
    };
} // namespace Ifrit::Test

void RpoolTest()
{
    TPooledConcurrentQueue<Ifrit::Test::Cat> q;
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
    FTaskScheduler  scheduler(8);
    Vec<TaskHandle> tasks;
    IF_LOG_INFO("DemoTest", "Starting task test");
    for (int i = 0; i < 100; i++)
    {
        auto task = scheduler.EnqueueTask(
            [&scheduler, &tasks, i](Task* task, void*) {
                printf("Task %lld is running\n", task->GetId());
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

void vectorTest() { auto p = Ifrit::GeometryProc::Sampler::LoadZpcPoissonSamplerReferences(); }

void vdbTest()
{
    auto vdbFileData = Ifrit::ReadBinaryFile("E:/bunny.vdb");
    auto vdbDesc     = Ifrit::GeometryProc::VDB::LoadVdbFromString(vdbFileData);
    Ifrit::GeometryProc::VDB::PrintVdbMeta(vdbDesc);
    auto p = Ifrit::GeometryProc::VDB::PoissonSampleVdbZpcReference(vdbDesc, 0.25f, 8);
    std::cout << "Sampled " << p.size() << " points from VDB." << std::endl;
    Ifrit::GeometryProc::PointCloud::PointCloudDescriptor pcDesc;
    pcDesc.m_Points = p.data();
    pcDesc.m_Count  = static_cast<u32>(p.size());

    Ifrit::GeometryProc::PointCloud::MoveCenterTo(pcDesc, Vector3f(32.0f, 32.0f, 32.0f));
    Ifrit::GeometryProc::PointCloud::NormalizeToLongestAxisAABB(pcDesc, Vector3f(0.0f), Vector3f(64.0f));

    // write points to "D:/vdb_points.txt"
    std::ofstream outFile("D:/vdb_points.txt");
    if (outFile.is_open())
    {
        for (const auto& point : p)
        {
            outFile << point.x << " " << point.y << " " << point.z << "\n";
        }
        outFile.close();
        IF_LOG_INFO("DemoTest", "Saved sampled points to D:/vdb_points.txt");
    }
    else
    {
        IF_LOG_INFO("DemoTest", "Failed to open file for writing sampled points.");
    }
}

int main()
{
    vdbTest();
    return 0;
}