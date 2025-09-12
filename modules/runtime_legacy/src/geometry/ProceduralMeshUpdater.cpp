#include "ifrit/runtime/geometry/ProceduralMeshUpdater.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/scene/SceneManager.h"
#include "ifrit/runtime/base/MeshComponent.h"
#include "ifrit/runtime/geometry/ProceduralMesh.h"
namespace Ifrit::Runtime::Geometry
{
    struct ProceduralMeshUpdaterInternalData
    {
        Owner<FrameGraphCompiler>   mFgCompiler;
        Owner<FrameGraphExecutor>   mFgExecutor;
        Ref<FrameGraphResourcePool> mResourcePool;
    };

    ProceduralMeshUpdater::ProceduralMeshUpdater() : mData(new ProceduralMeshUpdaterInternalData())
    {
        mData->mResourcePool = MakeRef<FrameGraphResourcePool>(GetActiveApplication()->GetRhi());
        mData->mFgExecutor   = MakeOwner<FrameGraphExecutor>(GetActiveApplication()->GetRhi());
        mData->mFgCompiler   = MakeOwner<FrameGraphCompiler>();
    }

    ProceduralMeshUpdater::~ProceduralMeshUpdater() { delete mData; }

    void                          ProceduralMeshUpdater::OnInitialize(IApplication* app) {}

    void                          ProceduralMeshUpdater::OnShutdown() {}

    void                          ProceduralMeshUpdater::OnFrameBegin() {}

    void                          ProceduralMeshUpdater::OnFrameEnd() {}

    Owner<RHI::RhiTaskSubmission> ProceduralMeshUpdater::OnPreRendering(RHI::RhiTaskSubmission* prevSubmission)
    {

        auto rhi   = GetActiveApplication()->GetRhi();
        auto queue = rhi->GetQueue(RHI::RhiQueueCapability::RhiQueue_Graphics);
        auto task  = queue->RunAsyncCommand(
            [&](const RHI::RhiCommandList* cmdList) {
                FrameGraphBuilder builder(GetActiveApplication()->GetShaderRegistry(), GetActiveApplication()->GetRhi(),
                     mData->mResourcePool.get());
                auto              scene = GetActiveApplication()->GetSceneManager()->GetActiveScene();
                auto              validObjs =
                    scene->FilterObjects([](GameObject* obj) { return obj->GetComponent<MeshFilter>() != nullptr; });
                for (auto obj : validObjs)
                {
                    auto meshFilter = obj->GetComponent<MeshFilter>();
                    if (meshFilter && meshFilter->IsEnabled())
                    {
                        auto mesh = meshFilter->GetMesh();
                        if (mesh && dynamic_cast<ProceduralMesh*>(mesh))
                        {
                            auto proceduralMesh = dynamic_cast<ProceduralMesh*>(mesh);
                            proceduralMesh->UpdateMesh(builder);
                        }
                    }
                }

                auto compiledGraph = mData->mFgCompiler->Compile(builder);
                mData->mFgExecutor->ExecuteInSingleCmd(cmdList, compiledGraph);
            },
            { prevSubmission }, {});
        return task;
    }

    Owner<RHI::RhiTaskSubmission> ProceduralMeshUpdater::OnPostRendering(RHI::RhiTaskSubmission* prevSubmission)
    {
        return nullptr;
    }

    void ProceduralMeshUpdater::OnUpdate(Scene* scene) {}
} // namespace Ifrit::Runtime::Geometry