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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "ifrit/runtime/physics/artemis/ArtemisSimulator.h"
#include "ifrit/runtime/physics/artemis/ArtemisSceneCollecting.h"

using namespace Ifrit::Math;
using namespace Ifrit::RHI;

namespace Ifrit::Runtime::Artemis
{
    struct ArtemisSimulatorPrivateData
    {
        Owner<FrameGraphCompiler>   m_FgCompiler;
        Owner<FrameGraphExecutor>   m_FgExecutor;
        Ref<FrameGraphResourcePool> m_ResourcePool;
        Vec<IArtemisSolver*>        m_Solvers;
    };

    ArtemisSimulator::ArtemisSimulator(IApplication* app) : m_App(app)
    {
        m_Data                 = new ArtemisSimulatorPrivateData();
        m_Data->m_ResourcePool = MakeRef<FrameGraphResourcePool>(m_App->GetRhi());
        m_Data->m_FgExecutor   = MakeOwner<FrameGraphExecutor>(m_App->GetRhi());
        m_Data->m_FgCompiler   = MakeOwner<FrameGraphCompiler>();
    }

    ArtemisSimulator::~ArtemisSimulator()
    {
        if (m_Data)
        {
            delete m_Data;
            m_Data = nullptr;
        }
    }

    IFRIT_APIDECL void ArtemisSimulator::CollectScene(Scene* scene, u32 frameId)
    {
        CollectPhysicsSceneData(scene, m_App->GetRhi(), frameId);
        for (auto& solver : m_Data->m_Solvers)
        {
            solver->CollectScene(scene);
        }
    }

    IFRIT_APIDECL Owner<RHI::RhiTaskSubmission> ArtemisSimulator::Update(
        f32 deltaTime, Vec<RHI::RhiTaskSubmission*> waitFor)
    {
        auto rhi   = m_App->GetRhi();
        auto queue = rhi->GetQueue(RhiQueueCapability::RhiQueue_Graphics);
        auto task  = queue->RunAsyncCommand(
            [&](const RhiCommandList* cmdList) {
                FrameGraphBuilder builder(m_App->GetShaderRegistry(), m_App->GetRhi(), m_Data->m_ResourcePool.get());
                for (auto& solver : m_Data->m_Solvers)
                {
                    solver->RunSolverStep(builder, deltaTime);
                }
                auto compiledGraph = m_Data->m_FgCompiler->Compile(builder);
                m_Data->m_FgExecutor->ExecuteInSingleCmd(cmdList, compiledGraph);
            },
            waitFor, {});
        return task;
    }

    void ArtemisSimulator::RegisterSolver(IArtemisSolver* solver)
    {
        if (solver)
        {
            m_Data->m_Solvers.push_back(solver);
        }
    }
} // namespace Ifrit::Runtime::Artemis
