#include "ifrit/runtime/physics/artemis/rigid/RigidSimulator.h"
#include "ifrit.internal/runtime/physics/artemis/InternalConst.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/physics/artemis/ArtemisSceneData.h"
#include "ifrit/runtime/rendercore/framegraph/FrameGraphUtils.h"
#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Artemis.h"

using namespace Ifrit::Runtime::FrameGraphUtils;
using namespace Ifrit::Runtime;

namespace Ifrit::Runtime::Artemis
{
    struct RigidSimulatorPrivateData
    {
        Scene*           m_ActiveScene = nullptr;
        RigidSimulator*  m_Parent      = nullptr;
        RigidBaseConfig* m_Config      = nullptr;

        FGBufferNodeRef  m_RDGColliderDataBuffer;
        FGBufferNodeRef  m_RDGColliderDataRuntimeBuffer;

        void             RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime);
        void             MotionTest(FrameGraphBuilder& builder, ArtemisSceneData* sceneData, f32 deltaTime);
    };

    void RigidSimulatorPrivateData::MotionTest(FrameGraphBuilder& builder, ArtemisSceneData* sceneData, f32 deltaTime)
    {
        struct PushConst
        {
            Vector4f        m_Gravity;
            f32             m_TimeStep;
            u32             m_NumColliders;
            RHI::RhiSRVDesc m_ColliderDataSRV;
            RHI::RhiUAVDesc m_ColliderDataRuntimeUAV;
        } pc;
        pc.m_Gravity      = Vector4f(0.0f, -1.0f, 0.0f, 0.0f);
        pc.m_NumColliders = sceneData->m_NumGpuColliders;
        pc.m_TimeStep     = deltaTime;

        auto numRigids = sceneData->m_NumGpuColliders;
        auto numTGX    = Math::DivRoundUp(numRigids, IfritShader::Artemis::Rigid::kRigidTGSizeX);

        AddComputePass<PushConst>(builder, "RigidSimulator.MotionTest",
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.RigidMotionTestCS, {}), Vector3i(numTGX, 1, 1),
            pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ColliderDataSRV        = ctx.m_FgDesc->GetSRV(*m_RDGColliderDataBuffer);
                pc.m_ColliderDataRuntimeUAV = ctx.m_FgDesc->GetUAV(*m_RDGColliderDataRuntimeBuffer);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGColliderDataBuffer);
    }

    void RigidSimulatorPrivateData::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        if (m_ActiveScene == nullptr || m_Config == nullptr)
        {
            IF_LOG_CRITICAL("RigidSimulator", "Active scene or config is not set.");
        }
        auto physicsData = CheckedPointerCast<ArtemisSceneData>(
            m_ActiveScene->GetPerFrameData()->m_ExtraData[Internal::kArtemisSceneDataKey]);

        m_RDGColliderDataBuffer = &builder.ImportBuffer(
            "ArtemisColliderDataBuffer", physicsData->m_GpuColliderDataBuffer[physicsData->m_FrameId % 2].get());
        m_RDGColliderDataRuntimeBuffer = &builder.ImportBuffer(
            "ArtemisColliderDataBufferRuntime", physicsData->m_GpuColliderDataBufferRuntime.get());

        MotionTest(builder, physicsData.get(), deltaTime);
    }

    IFRIT_APIDECL RigidSimulator::RigidSimulator() : m_Data(new RigidSimulatorPrivateData())
    {
        m_Data->m_Parent = this;
        m_Data->m_Config = &m_Config;
    }
    IFRIT_APIDECL RigidSimulator::~RigidSimulator()
    {
        delete m_Data;
        m_Data = nullptr;
    }

    IFRIT_APIDECL void RigidSimulator::SetConfig(const RigidBaseConfig& cfg)
    {
        m_Config         = cfg;
        m_Data->m_Config = &m_Config;
    }
    IFRIT_APIDECL RigidBaseConfig& RigidSimulator::GetActiveConfig() { return m_Config; }

    IFRIT_APIDECL void             RigidSimulator::CollectScene(Scene* scene)
    {
        m_Data->m_ActiveScene = scene;
        IF_LOG_ASSERTION("RigidSimulator",
            scene->GetPerFrameData()->m_ExtraData.count(Internal::kArtemisSceneDataKey) > 0,
            "ArtemisSceneData not found in scene's per-frame data");
    }

    IFRIT_APIDECL void RigidSimulator::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        m_Data->RunSolverStep(builder, deltaTime);
    }

} // namespace Ifrit::Runtime::Artemis
