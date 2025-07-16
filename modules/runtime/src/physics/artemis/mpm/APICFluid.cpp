#include "ifrit/runtime/physics/artemis/mpm/APICFluid.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Artemis.h"

#include "ifrit.shader.neo/Artemis/APIC/APICFluid.Utility.hlsli"

using namespace Ifrit::Runtime;
using namespace Ifrit::RHI;
using namespace Ifrit::Math;
using namespace Ifrit::Runtime::FrameGraphUtils;
using namespace Ifrit::Runtime::Internal;

namespace Ifrit::Runtime::Artemis
{
    struct APICFluidPrivateData
    {
        u32             m_GridSize          = 128;
        u32             m_NumParticles      = 180000;
        f32             m_DefaultTimestep   = 0.03f;
        f32             m_Density           = 0.001f;
        bool            m_Initialized       = false;
        u32             m_DefaultIterations = 200;
        u32             m_FrameId           = 0;

        /* Grid States
            BIT:0 Grid occupied, BIT:1-4 Boundary type, BIT:5-7 Weights
        */
        RhiBufferRef    m_GridStates;      // (R32_UINT)
        RhiBufferRef    m_GridVelocities;  // (R32G32_FLOAT)
        RhiBufferRef    m_GridMasses;      // (R32_FLOAT)
        RhiBufferRef    m_GridPressure[2]; // (R32_FLOAT)
        RhiBufferRef    m_GridCoefB;       // (R32_FLOAT) Ax = B

        RhiBufferRef    m_ParticleLocation; // (R32G32_FLOAT)
        RhiBufferRef    m_ParticleVelocity; // (R32G32_FLOAT)
        RhiBufferRef    m_ParticleMass;     // (R32_FLOAT)
        RhiBufferRef    m_ParticleC;        // (R32G32B32A32_FLOAT)
        RhiBufferRef    m_ParticleIndex;    // (R32_UINT)

        FGBufferNodeRef m_RDGGridStates;
        FGBufferNodeRef m_RDGGridVelocities;
        FGBufferNodeRef m_RDGGridMasses;
        FGBufferNodeRef m_RDGGridPressure[2];
        FGBufferNodeRef m_RDGGridCoefB;

        FGBufferNodeRef m_RDGParticleLocation;
        FGBufferNodeRef m_RDGParticleVelocity;
        FGBufferNodeRef m_RDGParticleMass;
        FGBufferNodeRef m_RDGParticleC;
        FGBufferNodeRef m_RDGParticleIndex;

        void            PrepareGPUResources(FrameGraphBuilder& builder);
        void            PrepareRDGResources(FrameGraphBuilder& builder);
        void            ParticleInit(FrameGraphBuilder& builder);
        void            ParticleToGrid(FrameGraphBuilder& builder);
        void            GridUpdate(FrameGraphBuilder& builder);
        void            GridReset(FrameGraphBuilder& builder);
        void            GridProjection(FrameGraphBuilder& builder, u32 numIterations);
        void            GridToParticle(FrameGraphBuilder& builder);
        void            ParticleAdvection(FrameGraphBuilder& builder);

        void            GridProjection_Precompute(FrameGraphBuilder& builder);
        void            GridProjection_Solve(FrameGraphBuilder& builder, u32 curIter);
        void            GridProjection_Apply(FrameGraphBuilder& builder);

        void            Render(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
    };

    void APICFluidPrivateData::PrepareGPUResources(FrameGraphBuilder& builder)
    {
        auto rhi              = builder.GetRhi();
        auto numGrids         = m_GridSize * m_GridSize;
        auto numParticles     = m_NumParticles;
        auto bufferUsage      = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
        auto bufferUsageIndex = bufferUsage | RhiBufferUsage::RhiBufferUsage_Index;
        auto bufferSizeBaseG  = sizeof(u32) * numGrids;
        auto bufferSizeBaseP  = sizeof(f32) * numParticles;

        m_GridStates      = rhi->CreateBufferDevice("APICFluid.GridStates", bufferSizeBaseG, bufferUsage, true);
        m_GridVelocities  = rhi->CreateBufferDevice("APICFluid.GridVelocities", bufferSizeBaseG * 2, bufferUsage, true);
        m_GridMasses      = rhi->CreateBufferDevice("APICFluid.GridMasses", bufferSizeBaseG, bufferUsage, true);
        m_GridPressure[0] = rhi->CreateBufferDevice("APICFluid.GridPressure0", bufferSizeBaseG, bufferUsage, true);
        m_GridPressure[1] = rhi->CreateBufferDevice("APICFluid.GridPressure1", bufferSizeBaseG, bufferUsage, true);
        m_GridCoefB       = rhi->CreateBufferDevice("APICFluid.GridCoefB", bufferSizeBaseG, bufferUsage, true);

        m_ParticleLocation =
            rhi->CreateBufferDevice("APICFluid.ParticleLocation", bufferSizeBaseP * 2, bufferUsage, true);
        m_ParticleVelocity =
            rhi->CreateBufferDevice("APICFluid.ParticleVelocity", bufferSizeBaseP * 2, bufferUsage, true);
        m_ParticleMass  = rhi->CreateBufferDevice("APICFluid.ParticleMass", bufferSizeBaseP, bufferUsage, true);
        m_ParticleC     = rhi->CreateBufferDevice("APICFluid.ParticleC", bufferSizeBaseP * 4, bufferUsage, true);
        m_ParticleIndex = rhi->CreateBufferDevice("APICFluid.ParticleIndex", bufferSizeBaseP, bufferUsageIndex, true);
    }

    void APICFluidPrivateData::PrepareRDGResources(FrameGraphBuilder& builder)
    {
        if (!m_Initialized)
        {
            PrepareGPUResources(builder);
            m_Initialized = true;
        }
        m_RDGGridStates      = &builder.ImportBuffer("APICFluid.GridStates", m_GridStates.get());
        m_RDGGridVelocities  = &builder.ImportBuffer("APICFluid.GridVelocities", m_GridVelocities.get());
        m_RDGGridMasses      = &builder.ImportBuffer("APICFluid.GridMasses", m_GridMasses.get());
        m_RDGGridPressure[0] = &builder.ImportBuffer("APICFluid.GridPressure0", m_GridPressure[0].get());
        m_RDGGridPressure[1] = &builder.ImportBuffer("APICFluid.GridPressure1", m_GridPressure[1].get());
        m_RDGGridCoefB       = &builder.ImportBuffer("APICFluid.GridCoefB", m_GridCoefB.get());

        m_RDGParticleLocation = &builder.ImportBuffer("APICFluid.ParticleLocation", m_ParticleLocation.get());
        m_RDGParticleVelocity = &builder.ImportBuffer("APICFluid.ParticleVelocity", m_ParticleVelocity.get());
        m_RDGParticleMass     = &builder.ImportBuffer("APICFluid.ParticleMass", m_ParticleMass.get());
        m_RDGParticleC        = &builder.ImportBuffer("APICFluid.ParticleC", m_ParticleC.get());
        m_RDGParticleIndex    = &builder.ImportBuffer("APICFluid.ParticleIndex", m_ParticleIndex.get());
    }

    void APICFluidPrivateData::ParticleInit(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_GridSize;
            u32 m_NumParticles;
            u32 m_ParticleLocationId;
            u32 m_ParticleVelocityId;
            u32 m_ParticleMassId;
            u32 m_ParticleIndexId;
            u32 m_ParticleCId;
        } pc;

        pc.m_GridSize           = m_GridSize;
        pc.m_NumParticles       = m_NumParticles;
        pc.m_ParticleLocationId = 0;
        pc.m_ParticleVelocityId = 0;
        pc.m_ParticleMassId     = 0;
        pc.m_ParticleIndexId    = 0;
        pc.m_ParticleCId        = 0;

        i32 tgX = DivRoundUp(m_NumParticles, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.ParticleInit",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidParticleInitCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleLocationId = ctx.m_FgDesc->GetUAV(*m_RDGParticleLocation);
                pc.m_ParticleVelocityId = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleMassId     = ctx.m_FgDesc->GetUAV(*m_RDGParticleMass);
                pc.m_ParticleIndexId    = ctx.m_FgDesc->GetUAV(*m_RDGParticleIndex);
                pc.m_ParticleCId        = ctx.m_FgDesc->GetUAV(*m_RDGParticleC);
                SetRootConstant(pc, ctx);
            })
            .AddWriteResource(*m_RDGParticleLocation)
            .AddWriteResource(*m_RDGParticleVelocity)
            .AddWriteResource(*m_RDGParticleIndex)
            .AddWriteResource(*m_RDGParticleC)
            .AddWriteResource(*m_RDGParticleMass);
    }

    void APICFluidPrivateData::GridReset(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_GridSize;
            u32 m_GridVelocityId;
            u32 m_GridStateId;
            u32 m_GridMassId;
        } pc;

        pc.m_GridSize       = m_GridSize;
        pc.m_GridVelocityId = 0;
        pc.m_GridStateId    = 0;
        pc.m_GridMassId     = 0;

        u32 numGrids = m_GridSize * m_GridSize;
        i32 tgX      = DivRoundUp((i32)numGrids, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.GridReset",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidGridResetCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_GridVelocityId = ctx.m_FgDesc->GetUAV(*m_RDGGridVelocities);
                pc.m_GridStateId    = ctx.m_FgDesc->GetUAV(*m_RDGGridStates);
                pc.m_GridMassId     = ctx.m_FgDesc->GetUAV(*m_RDGGridMasses);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGGridStates)
            .AddReadWriteResource(*m_RDGGridVelocities)
            .AddReadWriteResource(*m_RDGGridMasses);
    }

    void APICFluidPrivateData::ParticleToGrid(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_GridSize;
            u32 m_NumParticles;
            u32 m_GridVelocityId;
            u32 m_GridMassId;
            u32 m_ParticlePositionsId;
            u32 m_ParticleMassId;
            u32 m_ParticleVelocityId;
            u32 m_ParticleCId;
            u32 m_GridStateId;
        } pc;

        pc.m_GridSize            = m_GridSize;
        pc.m_NumParticles        = m_NumParticles;
        pc.m_GridVelocityId      = 0;
        pc.m_GridMassId          = 0;
        pc.m_ParticlePositionsId = 0;
        pc.m_ParticleMassId      = 0;
        pc.m_ParticleVelocityId  = 0;
        pc.m_ParticleCId         = 0;
        pc.m_GridStateId         = 0;

        i32 tgX = DivRoundUp(m_NumParticles, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.ParticleToGrid",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidP2GCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_GridVelocityId      = ctx.m_FgDesc->GetUAV(*m_RDGGridVelocities);
                pc.m_GridMassId          = ctx.m_FgDesc->GetUAV(*m_RDGGridMasses);
                pc.m_ParticlePositionsId = ctx.m_FgDesc->GetUAV(*m_RDGParticleLocation);
                pc.m_ParticleMassId      = ctx.m_FgDesc->GetUAV(*m_RDGParticleMass);
                pc.m_ParticleVelocityId  = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleCId         = ctx.m_FgDesc->GetUAV(*m_RDGParticleC);
                pc.m_GridStateId         = ctx.m_FgDesc->GetUAV(*m_RDGGridStates);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleLocation)
            .AddReadResource(*m_RDGParticleMass)
            .AddReadResource(*m_RDGParticleVelocity)
            .AddReadResource(*m_RDGParticleC)
            .AddReadWriteResource(*m_RDGGridStates)
            .AddReadWriteResource(*m_RDGGridVelocities)
            .AddReadWriteResource(*m_RDGGridMasses);
    }

    void APICFluidPrivateData::GridUpdate(FrameGraphBuilder& builder)
    {
        // TODO
        struct PushConst
        {
            Vector2f m_DefaultGravity;
            u32      m_NumGrids;
            u32      m_GridSize;
            f32      m_DefaultTimestep;
            u32      m_GridVelocitiesId;
            u32      m_GridMassId;
            u32      m_GridStateId;
        } pc;

        pc.m_DefaultGravity   = Vector2f(0.0f, 9.81f);
        pc.m_NumGrids         = m_GridSize * m_GridSize;
        pc.m_GridSize         = m_GridSize;
        pc.m_DefaultTimestep  = m_DefaultTimestep;
        pc.m_GridVelocitiesId = 0;
        pc.m_GridMassId       = 0;
        pc.m_GridStateId      = 0;

        i32 tgX = DivRoundUp((i32)pc.m_NumGrids, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.GridUpdate",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidGridUpdateCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_GridVelocitiesId = ctx.m_FgDesc->GetUAV(*m_RDGGridVelocities);
                pc.m_GridMassId       = ctx.m_FgDesc->GetUAV(*m_RDGGridMasses);
                pc.m_GridStateId      = ctx.m_FgDesc->GetUAV(*m_RDGGridStates);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGGridStates)
            .AddReadWriteResource(*m_RDGGridVelocities)
            .AddReadWriteResource(*m_RDGGridMasses);
    }

    void APICFluidPrivateData::GridProjection_Precompute(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            f32 m_Timestep;
            f32 m_Density;
            u32 m_GridSize;

            u32 m_GridVelocityId;
            u32 m_GridStateId;
            u32 m_GridCoefBId;
            u32 m_GridPressureId;
            u32 m_ClearPressure;
        } pc;

        pc.m_Timestep       = m_DefaultTimestep;
        pc.m_Density        = m_Density;
        pc.m_GridSize       = m_GridSize;
        pc.m_GridVelocityId = 0;
        pc.m_GridStateId    = 0;
        pc.m_GridCoefBId    = 0;
        pc.m_GridPressureId = 0;
        if (m_FrameId == 0)
        {
            pc.m_ClearPressure = 1; // Clear pressure on first frame
        }
        else
        {
            pc.m_ClearPressure = 0;
        }

        u32 numGrids = m_GridSize * m_GridSize;
        i32 tgX      = DivRoundUp((i32)numGrids, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.GridProjection.Precompute",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidGridProjectionSolveVelPrecomputeCS, {}),
            Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_GridVelocityId = ctx.m_FgDesc->GetUAV(*m_RDGGridVelocities);
                pc.m_GridStateId    = ctx.m_FgDesc->GetUAV(*m_RDGGridStates);
                pc.m_GridCoefBId    = ctx.m_FgDesc->GetUAV(*m_RDGGridCoefB);
                pc.m_GridPressureId = ctx.m_FgDesc->GetUAV(*m_RDGGridPressure[0]);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGGridStates)
            .AddReadWriteResource(*m_RDGGridVelocities)
            .AddReadWriteResource(*m_RDGGridCoefB)
            .AddReadWriteResource(*m_RDGGridPressure[0]);
    }

    void APICFluidPrivateData::GridProjection_Solve(FrameGraphBuilder& builder, u32 curIter)
    {
        struct PushConst
        {
            u32 m_GridSize;
            u32 m_GridCoefBId;
            u32 m_GridStateId;
            u32 m_GridLastPressureId;
            u32 m_GridCurPressureId;
        } pc;

        pc.m_GridSize           = m_GridSize;
        pc.m_GridCoefBId        = 0;
        pc.m_GridStateId        = 0;
        pc.m_GridLastPressureId = 0;
        pc.m_GridCurPressureId  = 0;

        u32 numGrids = m_GridSize * m_GridSize;
        i32 tgX      = DivRoundUp((i32)numGrids, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.GridProjection.Solve",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidGridProjectionSolveCS, {}), Vector3i(tgX, 1, 1), pc,
            [this, curIter](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_GridCoefBId        = ctx.m_FgDesc->GetUAV(*m_RDGGridCoefB);
                pc.m_GridStateId        = ctx.m_FgDesc->GetUAV(*m_RDGGridStates);
                pc.m_GridLastPressureId = ctx.m_FgDesc->GetUAV(*m_RDGGridPressure[curIter % 2]);
                pc.m_GridCurPressureId  = ctx.m_FgDesc->GetUAV(*m_RDGGridPressure[(curIter + 1) % 2]);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGGridStates)
            .AddReadWriteResource(*m_RDGGridCoefB)
            .AddReadWriteResource(*m_RDGGridPressure[curIter % 2])
            .AddReadWriteResource(*m_RDGGridPressure[(curIter + 1) % 2]);
    }

    void APICFluidPrivateData::GridProjection_Apply(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_GridSize;
            f32 m_Timestep;
            f32 m_Density;

            u32 m_GridVelocitiesId;
            u32 m_GridPressureId;
            u32 m_GridStateId;
        } pc;

        pc.m_GridSize         = m_GridSize;
        pc.m_Timestep         = m_DefaultTimestep;
        pc.m_Density          = m_Density;
        pc.m_GridVelocitiesId = 0;
        pc.m_GridPressureId   = 0;
        pc.m_GridStateId      = 0;

        u32 numGrids = m_GridSize * m_GridSize;
        i32 tgX      = DivRoundUp((i32)numGrids, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.GridProjection.Apply",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidGridProjectionApplyCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_GridVelocitiesId = ctx.m_FgDesc->GetUAV(*m_RDGGridVelocities);
                pc.m_GridPressureId   = ctx.m_FgDesc->GetUAV(*m_RDGGridPressure[0]);
                pc.m_GridStateId      = ctx.m_FgDesc->GetUAV(*m_RDGGridStates);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGGridStates)
            .AddReadWriteResource(*m_RDGGridVelocities)
            .AddReadWriteResource(*m_RDGGridPressure[0]);
    }

    void APICFluidPrivateData::GridProjection(FrameGraphBuilder& builder, u32 numIterations)
    {
        GridProjection_Precompute(builder);
        for (u32 i = 0; i < numIterations; ++i)
        {
            GridProjection_Solve(builder, i * 2);
            GridProjection_Solve(builder, i * 2 + 1);
        }
        GridProjection_Apply(builder);
    }

    void APICFluidPrivateData::GridToParticle(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_GridSize;
            u32 m_NumParticles;

            u32 m_ParticlePositionsId;
            u32 m_GridVelocitiesId;
            u32 m_ParticleCId;
            u32 m_ParticleVelocityId;
        } pc;

        pc.m_GridSize     = m_GridSize;
        pc.m_NumParticles = m_NumParticles;

        pc.m_ParticlePositionsId = 0;
        pc.m_GridVelocitiesId    = 0;
        pc.m_ParticleCId         = 0;
        pc.m_ParticleVelocityId  = 0;

        u32 numGrids = m_GridSize * m_GridSize;
        i32 tgX      = DivRoundUp(m_NumParticles, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.GridToParticle",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidG2PCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticlePositionsId = ctx.m_FgDesc->GetUAV(*m_RDGParticleLocation);
                pc.m_GridVelocitiesId    = ctx.m_FgDesc->GetUAV(*m_RDGGridVelocities);
                pc.m_ParticleCId         = ctx.m_FgDesc->GetUAV(*m_RDGParticleC);
                pc.m_ParticleVelocityId  = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGGridVelocities)
            .AddReadWriteResource(*m_RDGParticleLocation)
            .AddReadWriteResource(*m_RDGParticleC)
            .AddReadWriteResource(*m_RDGParticleVelocity);
    }

    void APICFluidPrivateData::ParticleAdvection(FrameGraphBuilder& builder)
    {

        struct PushConst
        {
            u32 m_GridSize;
            u32 m_NumParticles;
            f32 m_Timestep;

            u32 m_ParticleLocationId;
            u32 m_ParticleVelocityId;
        } pc;

        pc.m_GridSize     = m_GridSize;
        pc.m_NumParticles = m_NumParticles;
        pc.m_Timestep     = m_DefaultTimestep;

        pc.m_ParticleLocationId = 0;
        pc.m_ParticleVelocityId = 0;

        i32 tgX = DivRoundUp(m_NumParticles, IfritShader::Artemis::kArtemisTGSizeX);

        AddComputePass<PushConst>(builder, "APICFluid.ParticleAdvection",
            ShaderVariantDesc(kIntShaderTableArtemis.APICFluidParticleUpdateCS, {}), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleLocationId = ctx.m_FgDesc->GetUAV(*m_RDGParticleLocation);
                pc.m_ParticleVelocityId = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGParticleLocation)
            .AddReadWriteResource(*m_RDGParticleVelocity);
    }

    void APICFluidPrivateData::Render(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        struct PushConst
        {
            u32 m_PositionId;
            f32 m_GridRange;
            f32 m_AspectRatio;
        } pc;

        pc.m_PositionId = 0;
        pc.m_GridRange  = m_GridSize * m_GridSize;

        auto& pass = builder.AddGraphicsPass("APICFluid.Draw",
            ShaderVariantDesc(Internal::kIntShaderTableArtemis.ParticleRender2dVS, {}),
            ShaderVariantDesc(Internal::kIntShaderTableArtemis.ParticleRender2dFS, {}), GetPushConstSize<PushConst>(),
            RhiRasterizerTopology::Point);

        pass.SetExecutionFunction([renderTarget, this](const FrameGraphPassContext& ctx) {
            auto      rt = renderTarget;

            auto      cmd      = ctx.m_CmdList;
            auto      rtWidth  = rt->GetWidth();
            auto      rtHeight = rt->GetHeight();

            PushConst pc;
            pc.m_PositionId  = ctx.m_FgDesc->GetUAV(*m_RDGParticleLocation);
            pc.m_GridRange   = m_GridSize;
            pc.m_AspectRatio = (f32)rtWidth / (f32)rtHeight;

            cmd->AttachIndexBuffer(m_ParticleIndex.get());
            cmd->SetCullMode(RhiCullMode::None);
            cmd->SetPushConst(&pc, 0, sizeof(PushConst));
            cmd->DrawIndexed(m_NumParticles, 1, 0, 0, 0);
        });

        pass.AddRenderTarget(*renderTarget)
            .AddReadResource(*m_RDGParticleLocation)
            .AddReadResource(*m_RDGParticleIndex);
    }

    IFRIT_APIDECL APICFluid::APICFluid() : m_Data(new APICFluidPrivateData()) {}
    IFRIT_APIDECL APICFluid::~APICFluid()
    {
        delete m_Data;
        m_Data = nullptr;
    }

    IFRIT_APIDECL void APICFluid::RunSolver(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        if (!m_Data->m_Initialized)
        {
            m_Data->PrepareRDGResources(builder);
            m_Data->ParticleInit(builder);
        }
        else
        {
            m_Data->PrepareRDGResources(builder);
        }

        m_Data->GridReset(builder);
        m_Data->ParticleToGrid(builder);
        m_Data->GridUpdate(builder);
        m_Data->GridProjection(builder, m_Data->m_DefaultIterations);
        m_Data->GridToParticle(builder);
        m_Data->ParticleAdvection(builder);
        m_Data->Render(builder, renderTarget);

        m_Data->m_FrameId++;
    }

} // namespace Ifrit::Runtime::Artemis