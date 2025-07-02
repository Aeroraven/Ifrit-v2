#include "ifrit/runtime/physics/siro/mpm/MPMSimulator.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Siro.h"
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit/core/math/linalg/LinalgOps.h"

using namespace Ifrit::Math;
using namespace Ifrit::Graphics::Rhi;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime::Siro
{
    template <u32 Dimension> struct MPMSimulatorTypes;

    template <> struct MPMSimulatorTypes<2>
    {
        using FSpatialVector           = Vector2f;
        using FSpatialTransform        = Matrix2x2f;
        using FSpatialVectorAligned    = Vector2f;
        using FSpatialTransformAligned = Matrix2x2f;
        using FScalar                  = f32;

        static IF_CONSTEXPR u32 kFSpatialVectorSize           = static_cast<u32>(sizeof(FSpatialVector));
        static IF_CONSTEXPR u32 kFSpatialTransformSize        = static_cast<u32>(sizeof(FSpatialTransform));
        static IF_CONSTEXPR u32 kFSpatialVectorAlignedSize    = static_cast<u32>(sizeof(FSpatialVectorAligned));
        static IF_CONSTEXPR u32 kFSpatialTransformAlignedSize = static_cast<u32>(sizeof(FSpatialTransformAligned));
        static IF_CONSTEXPR u32 kFScalarSize                  = static_cast<u32>(sizeof(FScalar));
    };

    template <> struct MPMSimulatorTypes<3>
    {
        using FSpatialVector           = Vector3f;
        using FSpatialTransform        = Matrix3x3f;
        using FSpatialVectorAligned    = Vector4f;
        using FSpatialTransformAligned = Matrixg<f32, 4, 3>;
        using FScalar                  = f32;

        static IF_CONSTEXPR u32 kFSpatialVectorSize           = static_cast<u32>(sizeof(FSpatialVector));
        static IF_CONSTEXPR u32 kFSpatialTransformSize        = static_cast<u32>(sizeof(FSpatialTransform));
        static IF_CONSTEXPR u32 kFSpatialVectorAlignedSize    = static_cast<u32>(sizeof(FSpatialVectorAligned));
        static IF_CONSTEXPR u32 kFSpatialTransformAlignedSize = static_cast<u32>(sizeof(FSpatialTransformAligned));
        static IF_CONSTEXPR u32 kFScalarSize                  = static_cast<u32>(sizeof(FScalar));
    };

    struct MPMSimulatorGridAttribute
    {
        Vector4u m_GridSize;
        Vector4u m_GridBoundaryWidth;
        Vector4f m_GridTranslation;

        u32      m_GridVelocity;
        u32      m_GridForce;
        u32      m_GridMass;
        f32      m_GridSpacing;
    };

    struct MPMSimulatorPrivateData
    {
        static IF_CONSTEXPR u32       kDefaultTGX = IfritShader::Siro::MPM::kMpmTGSizeX;

        MPMSimulatorConfig*           m_Config              = nullptr;
        bool                          m_RebuildGPUResources = true;

        // Persistent data
        RhiBufferRef                  m_ParticlePosition;
        RhiBufferRef                  m_ParticleVelocity;
        RhiBufferRef                  m_ParticleMass;
        RhiBufferRef                  m_ParticleDeformGrad;
        RhiBufferRef                  m_ParticleDeformGradDet;
        RhiBufferRef                  m_ParticleVolume;
        RhiBufferRef                  m_ParticleApicB;
        RhiBufferRef                  m_ParticleIndex;
        RhiBufferRef                  m_ParticleDebug;

        RhiBufferRef                  m_GridForce;
        RhiBufferRef                  m_GridVelocity;
        RhiBufferRef                  m_GridMass;

        RhiBufferRef                  m_GridAttribute;

        FGBufferNodeRef               m_RDGParticlePosition;
        FGBufferNodeRef               m_RDGParticleVelocity;
        FGBufferNodeRef               m_RDGParticleMass;
        FGBufferNodeRef               m_RDGParticleDeformGrad;
        FGBufferNodeRef               m_RDGParticleDeformGradDet;
        FGBufferNodeRef               m_RDGParticleVolume;
        FGBufferNodeRef               m_RDGParticleApicB;
        FGBufferNodeRef               m_RDGParticleDebug;
        FGBufferNodeRef               m_RDGGridForce;
        FGBufferNodeRef               m_RDGGridVelocity;
        FGBufferNodeRef               m_RDGGridMass;
        FGBufferNodeRef               m_RDGGridAttribute;

        // Transient data
        FGBufferNodeRef               m_RDGValidGridCounter;
        FGBufferNodeRef               m_RDGValidGridList;

        // Init
        template <u32 Dimension> void InitGPUResources(RhiBackend* RHI);
        void                          PrepareInitialGPUData(RhiBackend* RHI);
        void                          InitRDGResources(FrameGraphBuilder& builder);
        void                          RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime);
        ShaderVariantDesc             GetShader(const String& name) const;
        u32                           GetNumGrids() const;

        // Solver steps
        void                          ParticleInit(FrameGraphBuilder& builder);
        void                          GridReset(FrameGraphBuilder& builder);
        void                          ParticleToGridTransfer(FrameGraphBuilder& builder, f32 deltaTime);
        void                          GridVelocityNormalize(FrameGraphBuilder& builder);
        void                          GridForceUpdate(FrameGraphBuilder& builder);
        void                          GridGravityApply(FrameGraphBuilder& builder);
        void                          GridVelocityUpdate(FrameGraphBuilder& builder, f32 deltaTime);
        void                          GridToParticleTransfer(FrameGraphBuilder& builder, f32 deltaTime);
        void                          ParticleAdvect(FrameGraphBuilder& builder, f32 deltaTime);

        // Visualizer
        void                          ParticleRender2D(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
        void                          ParticleRender3D(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
    };

    void MPMSimulatorPrivateData::ParticleInit(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_NumParticles;
            f32 m_Mass;
            f32 m_Density;
            u32 m_Grid;
            u32 m_ParticleLocation;
            u32 m_ParticleVelocity;
            u32 m_ParticleMass;
            u32 m_ParticleVolume;
            u32 m_ParticleB;
            u32 m_ParticleDeformationGrad;
            u32 m_ParticleDeformationGradDet;
        } pc;
        pc.m_NumParticles               = m_Config->m_DefaultNumParticles;
        pc.m_Mass                       = m_Config->m_DefaultMass;
        pc.m_Density                    = m_Config->m_DefaultDensity;
        pc.m_Grid                       = 0;
        pc.m_ParticleLocation           = 0;
        pc.m_ParticleVelocity           = 0;
        pc.m_ParticleMass               = 0;
        pc.m_ParticleVolume             = 0;
        pc.m_ParticleB                  = 0;
        pc.m_ParticleDeformationGrad    = 0;
        pc.m_ParticleDeformationGradDet = 0;

        auto tgX = static_cast<i32>(DivRoundUp(pc.m_NumParticles, kDefaultTGX));

        AddComputePass<PushConst>(builder, "MPMSimulator.ParticleInit",
            GetShader(Internal::kIntShaderTableSiro.MPMParticleInitCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid                       = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleLocation           = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleVelocity           = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleMass               = ctx.m_FgDesc->GetUAV(*m_RDGParticleMass);
                pc.m_ParticleVolume             = ctx.m_FgDesc->GetUAV(*m_RDGParticleVolume);
                pc.m_ParticleB                  = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_ParticleDeformationGrad    = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleDeformationGradDet = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGradDet);

                SetRootConstant(pc, ctx);
            })
            .AddWriteResource(*m_RDGParticlePosition)
            .AddWriteResource(*m_RDGParticleVelocity)
            .AddWriteResource(*m_RDGParticleMass)
            .AddWriteResource(*m_RDGParticleVolume)
            .AddWriteResource(*m_RDGParticleApicB)
            .AddWriteResource(*m_RDGParticleDeformGrad)
            .AddWriteResource(*m_RDGParticleDeformGradDet)
            .AddReadResource(*m_RDGGridAttribute);
    }

    void MPMSimulatorPrivateData::GridReset(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_Grid;
        } pc;
        pc.m_Grid = 0;

        auto numGrids = GetNumGrids();
        auto tgX      = static_cast<i32>(DivRoundUp(numGrids, kDefaultTGX));

        AddComputePass<PushConst>(builder, "MPMSimulator.GridReset",
            GetShader(Internal::kIntShaderTableSiro.MPMGridResetCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                SetRootConstant(pc, ctx);
            })
            .AddWriteResource(*m_RDGGridAttribute);
    }

    void MPMSimulatorPrivateData::ParticleToGridTransfer(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            u32 m_NumParticles;
            f32 m_DeltaTime;
            u32 m_ParticleVelocity;
            u32 m_ParticleLocation;
            u32 m_ParticleMass;
            u32 m_ParticleB;
            u32 m_Grid;
            u32 m_ParticleDeformGrad;
            u32 m_ParticleDeformGradDet;
        } pc;

        pc.m_NumParticles          = m_Config->m_DefaultNumParticles;
        pc.m_DeltaTime             = deltaTime;
        pc.m_ParticleVelocity      = 0;
        pc.m_ParticleLocation      = 0;
        pc.m_ParticleMass          = 0;
        pc.m_ParticleB             = 0;
        pc.m_Grid                  = 0;
        pc.m_ParticleDeformGrad    = 0;
        pc.m_ParticleDeformGradDet = 0;

        auto tgX = static_cast<i32>(DivRoundUp(pc.m_NumParticles, kDefaultTGX));

        AddComputePass<PushConst>(builder, "MPMSimulator.ParticleToGridTransfer",
            GetShader(Internal::kIntShaderTableSiro.MPMP2GCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleVelocity      = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleLocation      = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleMass          = ctx.m_FgDesc->GetUAV(*m_RDGParticleMass);
                pc.m_ParticleB             = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_Grid                  = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleDeformGrad    = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleDeformGradDet = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGradDet);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleVelocity)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleMass)
            .AddReadResource(*m_RDGParticleApicB)
            .AddWriteResource(*m_RDGGridAttribute)
            .AddWriteResource(*m_RDGGridVelocity)
            .AddWriteResource(*m_RDGGridMass)
            .AddReadWriteResource(*m_RDGParticleDeformGradDet)
            .AddReadWriteResource(*m_RDGParticleDeformGrad);
    }

    void MPMSimulatorPrivateData::GridVelocityNormalize(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_Grid;
            u32 m_ValidGridCounter;
            u32 m_ValidGridList;
        } pc;

        pc.m_Grid             = 0;
        pc.m_ValidGridCounter = 0;
        pc.m_ValidGridList    = 0;

        auto numGrids = GetNumGrids();
        auto tgX      = static_cast<i32>(DivRoundUp(numGrids, kDefaultTGX));

        AddClearUAVPass(builder, "MPMSimulator.ClearValidGridCounter", *m_RDGValidGridCounter, 0)
            .AddWriteResource(*m_RDGValidGridCounter);

        AddComputePass<PushConst>(builder, "MPMSimulator.GridVelocityNormalize",
            GetShader(Internal::kIntShaderTableSiro.MPMGridRegularizeCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid             = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ValidGridCounter = ctx.m_FgDesc->GetUAV(*m_RDGValidGridCounter);
                pc.m_ValidGridList    = ctx.m_FgDesc->GetUAV(*m_RDGValidGridList);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGGridAttribute)
            .AddWriteResource(*m_RDGGridVelocity)
            .AddWriteResource(*m_RDGValidGridCounter)
            .AddWriteResource(*m_RDGValidGridList);
    }

    void MPMSimulatorPrivateData::GridForceUpdate(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            Vector4f m_MaterialParameters1;
            u32      m_NumParticles;
            u32      m_Grid;
            u32      m_ParticleLocation;
            u32      m_ParticleVolume;
            u32      m_ParticleDeformationGrad;
            u32      m_ParticleDeformationGradDet;
        } pc;
        auto Mu     = m_Config->m_DefaultNeoHookeanMu;
        auto Lambda = m_Config->m_DefaultNeoHookeanLambda;

        pc.m_MaterialParameters1        = Vector4f(Mu, Lambda, 0.0f, 0.0f);
        pc.m_NumParticles               = m_Config->m_DefaultNumParticles;
        pc.m_Grid                       = 0;
        pc.m_ParticleLocation           = 0;
        pc.m_ParticleVolume             = 0;
        pc.m_ParticleDeformationGrad    = 0;
        pc.m_ParticleDeformationGradDet = 0;

        auto tgX = static_cast<i32>(DivRoundUp(pc.m_NumParticles, kDefaultTGX));

        AddComputePass<PushConst>(builder, "MPMSimulator.GridForceUpdate",
            GetShader(Internal::kIntShaderTableSiro.MPMGridForceUpdateCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid                       = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleLocation           = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleVolume             = ctx.m_FgDesc->GetUAV(*m_RDGParticleVolume);
                pc.m_ParticleDeformationGrad    = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleDeformationGradDet = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGradDet);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGGridAttribute)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleVolume)
            .AddReadResource(*m_RDGParticleDeformGrad)
            .AddReadWriteResource(*m_RDGParticleDeformGradDet)
            .AddWriteResource(*m_RDGGridForce);
    }

    void MPMSimulatorPrivateData::GridGravityApply(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            Vector4f m_Gravity;
            u32      m_Grid;
            u32      m_ValidGridCounter;
            u32      m_ValidGridList;
        };

        PushConst pc;
        pc.m_Gravity          = Vector4f(m_Config->m_Gravity.xyz(), 0.0f);
        pc.m_Grid             = 0;
        pc.m_ValidGridCounter = 0;
        pc.m_ValidGridList    = 0;

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.GridGravityApply",
            GetShader(Internal::kIntShaderTableSiro.MPMGridGravityApplyCS), *m_RDGValidGridCounter, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid             = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ValidGridCounter = ctx.m_FgDesc->GetUAV(*m_RDGValidGridCounter);
                pc.m_ValidGridList    = ctx.m_FgDesc->GetUAV(*m_RDGValidGridList);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGGridAttribute)
            .AddReadResource(*m_RDGValidGridCounter)
            .AddReadResource(*m_RDGValidGridList)
            .AddReadWriteResource(*m_RDGGridForce)
            .AddReadResource(*m_RDGGridMass);
    }

    void MPMSimulatorPrivateData::GridVelocityUpdate(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            f32 m_DeltaTime;
            u32 m_Grid;
            u32 m_ValidGridCounter;
            u32 m_ValidGridList;
        } pc;

        pc.m_DeltaTime        = deltaTime;
        pc.m_Grid             = 0;
        pc.m_ValidGridCounter = 0;
        pc.m_ValidGridList    = 0;

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.GridVelocityUpdate",
            GetShader(Internal::kIntShaderTableSiro.MPMGridVelocityUpdateCS), *m_RDGValidGridCounter, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid             = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ValidGridCounter = ctx.m_FgDesc->GetUAV(*m_RDGValidGridCounter);
                pc.m_ValidGridList    = ctx.m_FgDesc->GetUAV(*m_RDGValidGridList);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGGridAttribute)
            .AddReadResource(*m_RDGGridForce)
            .AddReadResource(*m_RDGValidGridCounter)
            .AddReadResource(*m_RDGValidGridList)
            .AddReadResource(*m_RDGGridVelocity);
    }

    void MPMSimulatorPrivateData::ParticleAdvect(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            u32 m_NumParticles;
            f32 m_DeltaTime;
            u32 m_ParticleLocation;
            u32 m_ParticleVelocity;
        } pc;
        pc.m_NumParticles     = m_Config->m_DefaultNumParticles;
        pc.m_DeltaTime        = deltaTime;
        pc.m_ParticleLocation = 0;
        pc.m_ParticleVelocity = 0;

        auto tgX = static_cast<i32>(DivRoundUp(pc.m_NumParticles, kDefaultTGX));

        AddComputePass<PushConst>(builder, "MPMSimulator.ParticleAdvect",
            GetShader(Internal::kIntShaderTableSiro.MPMParticleAdvectionCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleLocation = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleVelocity = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleVelocity);
    }

    void MPMSimulatorPrivateData::PrepareInitialGPUData(RhiBackend* RHI)
    {
        auto     numParticles = m_Config->m_DefaultNumParticles;
        Vec<u32> particleIndexData(numParticles);
        for (u32 i = 0; i < numParticles; ++i)
            particleIndexData[i] = i;

        MPMSimulatorGridAttribute gridAttr;
        gridAttr.m_GridSize          = Vector4u(m_Config->m_GridSize.xyz(), 0);
        gridAttr.m_GridBoundaryWidth = Vector4u(m_Config->m_GridBoundaryWidth.xyz(), 0);
        gridAttr.m_GridTranslation   = Vector4f(m_Config->m_GridOffset.xyz(), 0.0f);

        gridAttr.m_GridVelocity = RHI->GetUAVDescriptor(m_GridVelocity.get());
        gridAttr.m_GridForce    = RHI->GetUAVDescriptor(m_GridForce.get());
        gridAttr.m_GridMass     = RHI->GetUAVDescriptor(m_GridMass.get());
        gridAttr.m_GridSpacing  = m_Config->m_GridSpacing;

        auto tq             = RHI->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
        auto stagedIndex    = RHI->CreateStagedSingleBuffer(m_ParticleIndex.get());
        auto stagedGridAttr = RHI->CreateStagedSingleBuffer(m_GridAttribute.get());
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            stagedIndex->CmdCopyToDevice(cmd, particleIndexData.data(), particleIndexData.size() * sizeof(u32), 0);
            stagedGridAttr->CmdCopyToDevice(cmd, &gridAttr, sizeof(MPMSimulatorGridAttribute), 0);
        });
    }

    void MPMSimulatorPrivateData::GridToParticleTransfer(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            i32 m_NumParticles;
            f32 m_DeltaTime;
            u32 m_Grid;
            u32 m_ParticleDeformationGrad;
            u32 m_ParticleLocation;
            u32 m_ParticleB;
            u32 m_ParticleVelocity;
        } pc;

        pc.m_NumParticles            = m_Config->m_DefaultNumParticles;
        pc.m_DeltaTime               = deltaTime;
        pc.m_Grid                    = 0;
        pc.m_ParticleDeformationGrad = 0;
        pc.m_ParticleLocation        = 0;
        pc.m_ParticleB               = 0;
        pc.m_ParticleVelocity        = 0;

        auto tgX = static_cast<i32>(DivRoundUp(pc.m_NumParticles, kDefaultTGX));

        AddComputePass<PushConst>(builder, "MPMSimulator.GridToParticleTransfer",
            GetShader(Internal::kIntShaderTableSiro.MPMG2PCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid                    = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleDeformationGrad = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleLocation        = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleB               = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_ParticleVelocity        = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGGridAttribute)
            .AddReadResource(*m_RDGGridVelocity)
            .AddReadResource(*m_RDGGridMass)
            .AddReadWriteResource(*m_RDGParticleDeformGrad)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadWriteResource(*m_RDGParticleApicB)
            .AddWriteResource(*m_RDGParticleVelocity);
    }

    u32 MPMSimulatorPrivateData::GetNumGrids() const
    {
        iAssertion(m_Config, "MPMSimulator: Config must be set before getting the number of grids.");
        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
            return m_Config->m_GridSize.x * m_Config->m_GridSize.y;
        else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
            return m_Config->m_GridSize.x * m_Config->m_GridSize.y * m_Config->m_GridSize.z;
    }

    template <u32 Dimension> void MPMSimulatorPrivateData::InitGPUResources(RhiBackend* RHI)
    {
        using MTypes = MPMSimulatorTypes<Dimension>;
        iAssertion(m_Config, "MPMSimulator: Config must be set before initializing GPU resources.");
        static_assert(Dimension == 2 || Dimension == 3, "MPMSimulator: Invalid dimension specified.");

        auto numParticles = m_Config->m_DefaultNumParticles;
        auto numGrids     = GetNumGrids();

        auto particlePosSz        = numParticles * MTypes::kFSpatialVectorAlignedSize;
        auto particleVelSz        = numParticles * MTypes::kFSpatialVectorAlignedSize;
        auto particleMassSz       = numParticles * MTypes::kFScalarSize;
        auto particleDeformGradSz = numParticles * MTypes::kFSpatialTransformAlignedSize;
        auto particleJSz          = numParticles * MTypes::kFScalarSize;
        auto particleVolSz        = numParticles * MTypes::kFScalarSize;
        auto particleApicBSz      = numParticles * MTypes::kFSpatialTransformAlignedSize;
        auto particleIndexSz      = numParticles * sizeof(u32);
        auto particleDebugSz      = numParticles * MTypes::kFSpatialVectorAlignedSize;

        auto gridForceSz = numGrids * MTypes::kFSpatialVectorAlignedSize;
        auto gridVelSz   = numGrids * MTypes::kFSpatialVectorAlignedSize;
        auto gridMassSz  = numGrids * MTypes::kFScalarSize;
        auto gridAttrSz  = sizeof(MPMSimulatorGridAttribute);

        auto defaultUsage = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
        auto indexUsage   = defaultUsage | RhiBufferUsage::RhiBufferUsage_Index;

        m_ParticlePosition = RHI->CreateBufferDevice("MPM_ParticlePosition", particlePosSz, defaultUsage, true);
        m_ParticleVelocity = RHI->CreateBufferDevice("MPM_ParticleVelocity", particleVelSz, defaultUsage, true);
        m_ParticleMass     = RHI->CreateBufferDevice("MPM_ParticleMass", particleMassSz, defaultUsage, true);
        m_ParticleDeformGrad =
            RHI->CreateBufferDevice("MPM_ParticleDeformGradient", particleDeformGradSz, defaultUsage, true);
        m_ParticleDeformGradDet =
            RHI->CreateBufferDevice("MPM_ParticleDeformGradientDeterminant", particleJSz, defaultUsage, true);
        m_ParticleVolume = RHI->CreateBufferDevice("MPM_ParticleVolume", particleVolSz, defaultUsage, true);
        m_ParticleApicB  = RHI->CreateBufferDevice("MPM_ParticleApicB", particleApicBSz, defaultUsage, true);
        m_ParticleIndex  = RHI->CreateBufferDevice("MPM_ParticleIndex", particleIndexSz, indexUsage, true);
        m_ParticleDebug  = RHI->CreateBufferDevice("MPM_ParticleDebug", particleDebugSz, defaultUsage, true);

        m_GridForce     = RHI->CreateBufferDevice("MPM_GridForce", gridForceSz, defaultUsage, true);
        m_GridVelocity  = RHI->CreateBufferDevice("MPM_GridVelocity", gridVelSz, defaultUsage, true);
        m_GridMass      = RHI->CreateBufferDevice("MPM_GridMass", gridMassSz, defaultUsage, true);
        m_GridAttribute = RHI->CreateBufferDevice("MPM_GridAttribute", gridAttrSz, defaultUsage, true);

        PrepareInitialGPUData(RHI);
    }

    void MPMSimulatorPrivateData::InitRDGResources(FrameGraphBuilder& builder)
    {
        // Persistent
        m_RDGParticlePosition   = &builder.ImportBuffer("MPM_ParticlePosition", m_ParticlePosition.get());
        m_RDGParticleVelocity   = &builder.ImportBuffer("MPM_ParticleVelocity", m_ParticleVelocity.get());
        m_RDGParticleMass       = &builder.ImportBuffer("MPM_ParticleMass", m_ParticleMass.get());
        m_RDGParticleDeformGrad = &builder.ImportBuffer("MPM_ParticleDeformGradient", m_ParticleDeformGrad.get());
        m_RDGParticleDeformGradDet =
            &builder.ImportBuffer("MPM_ParticleDeformGradientDeterminant", m_ParticleDeformGradDet.get());
        m_RDGParticleVolume = &builder.ImportBuffer("MPM_ParticleVolume", m_ParticleVolume.get());
        m_RDGParticleApicB  = &builder.ImportBuffer("MPM_ParticleApicB", m_ParticleApicB.get());
        m_RDGParticleDebug  = &builder.ImportBuffer("MPM_ParticleDebug", m_ParticleDebug.get());
        m_RDGGridForce      = &builder.ImportBuffer("MPM_GridForce", m_GridForce.get());
        m_RDGGridVelocity   = &builder.ImportBuffer("MPM_GridVelocity", m_GridVelocity.get());
        m_RDGGridMass       = &builder.ImportBuffer("MPM_GridMass", m_GridMass.get());
        m_RDGGridAttribute  = &builder.ImportBuffer("MPM_GridAttribute", m_GridAttribute.get());

        // Transient
        auto defaultUsage  = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
        auto indirectUsage = defaultUsage | RhiBufferUsage::RhiBufferUsage_Indirect;
        auto numGrids      = GetNumGrids();

        m_RDGValidGridCounter =
            &builder.DeclareBuffer("MPM_ValidGridCounter", FrameGraphBufferDesc(sizeof(u32) * 4, indirectUsage));
        m_RDGValidGridList =
            &builder.DeclareBuffer("MPM_ValidGridList", FrameGraphBufferDesc(sizeof(u32) * numGrids, defaultUsage));
    }

    void MPMSimulatorPrivateData::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        auto rhi        = builder.GetRhi();
        bool isFirstRun = m_RebuildGPUResources;
        if (m_RebuildGPUResources)
        {
            m_RebuildGPUResources = false;
            if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
                InitGPUResources<2>(rhi);
            else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
                InitGPUResources<3>(rhi);
            else
                iAssertion(false, "MPMSimulator: Invalid problem dimension specified.");
        }
        InitRDGResources(builder);
        if (isFirstRun)
        {
            ParticleInit(builder);
        }
        GridReset(builder);
        ParticleToGridTransfer(builder, deltaTime);
        GridVelocityNormalize(builder);
        GridForceUpdate(builder);
        GridGravityApply(builder);
        GridVelocityUpdate(builder, deltaTime);
        GridToParticleTransfer(builder, deltaTime);
        ParticleAdvect(builder, deltaTime);
    }

    ShaderVariantDesc MPMSimulatorPrivateData::GetShader(const String& name) const
    {

        Vec<String> shaderVariants;

        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
        {
            shaderVariants.push_back("IFSHADER_MPM_3D");
        }
        if (m_Config->m_Variant == MPMSimulatorVariant::MLS)
        {
            shaderVariants.push_back("IFSHADER_MPM_MLS");
        }
        return ShaderVariantDesc(name, shaderVariants);
    }

    void MPMSimulatorPrivateData::ParticleRender2D(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        struct PushConst
        {
            u32 m_PositionId;
            f32 m_GridRange;
            f32 m_AspectRatio;
        };

        auto& pass = builder.AddGraphicsPass("MPMSimulator.ParticleRender",
            ShaderVariantDesc(Internal::kIntShaderTableSiro.ParticleRender2dVS, {}),
            ShaderVariantDesc(Internal::kIntShaderTableSiro.ParticleRender2dFS, {}), GetPushConstSize<PushConst>(),
            RhiRasterizerTopology::Point);

        pass.SetExecutionFunction([renderTarget, this](const FrameGraphPassContext& ctx) {
            auto      rt = renderTarget;

            auto      cmd      = ctx.m_CmdList;
            auto      rtWidth  = rt->GetWidth();
            auto      rtHeight = rt->GetHeight();

            PushConst pc;
            pc.m_PositionId  = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
            pc.m_GridRange   = m_Config->m_GridSize.x * m_Config->m_GridSpacing;
            pc.m_AspectRatio = (f32)rtWidth / (f32)rtHeight;

            cmd->AttachIndexBuffer(m_ParticleIndex.get());
            cmd->SetCullMode(RhiCullMode::None);
            cmd->SetPushConst(&pc, 0, sizeof(PushConst));
            cmd->DrawIndexed(m_Config->m_DefaultNumParticles, 1, 0, 0, 0);
        });
        pass.AddRenderTarget(*renderTarget).AddReadResource(*m_RDGParticlePosition);
    }

    void MPMSimulatorPrivateData::ParticleRender3D(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        f32      camNear  = 0.1f;
        auto     rtWidth  = renderTarget->GetWidth();
        auto     rtHeight = renderTarget->GetHeight();

        Vector3f gridSizef =
            Vector3f(f32(m_Config->m_GridSize.x), f32(m_Config->m_GridSize.y), f32(m_Config->m_GridSize.z));
        Vector3f gridMinPos = m_Config->m_GridOffset;
        Vector3f gridMaxPos = m_Config->m_GridOffset + gridSizef * m_Config->m_GridSpacing;

        Vector3f gridCenter = (gridMinPos + gridMaxPos) * 0.5f;
        Vector3f gridExtent = (gridMaxPos - gridMinPos) * 0.5f;

        Vector3f cameraPos = gridCenter - Vector3f(0.0f, 0.0f, gridExtent.z * 2.0f);

        auto     lookAt      = Math::LookAt(cameraPos, gridCenter, Vector3f(0.0f, 1.0f, 0.0f));
        auto     fovyRad     = 60.0f / 180.0f * 3.14159265358979323846f;
        auto     aspectRatio = (f32)rtWidth / (f32)rtHeight;
        auto     proj        = Math::PerspectiveNegateY(fovyRad, aspectRatio, camNear, 1000.0f);
        auto     mvp         = Math::Transpose(Math::MatMul(proj, lookAt));

        struct PushConst
        {
            Matrix4x4f m_MVP;
            u32        m_PositionId;
        } pc;

        auto& pass = builder.AddGraphicsPass("MPMSimulator.ParticleRender3D",
            ShaderVariantDesc(Internal::kIntShaderTableSiro.ParticleRender3dVS, {}),
            ShaderVariantDesc(Internal::kIntShaderTableSiro.ParticleRender3dFS, {}), GetPushConstSize<PushConst>(),
            RhiRasterizerTopology::Point);

        pass.SetExecutionFunction([renderTarget, this, mvp](const FrameGraphPassContext& ctx) {
            auto      rt = renderTarget;

            auto      cmd      = ctx.m_CmdList;
            auto      rtWidth  = rt->GetWidth();
            auto      rtHeight = rt->GetHeight();

            PushConst pc;
            pc.m_PositionId = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
            pc.m_MVP        = mvp;

            cmd->AttachIndexBuffer(m_ParticleIndex.get());
            cmd->SetCullMode(RhiCullMode::None);
            cmd->SetPushConst(&pc, 0, sizeof(PushConst));
            cmd->DrawIndexed(m_Config->m_DefaultNumParticles, 1, 0, 0, 0);
        });
        pass.AddRenderTarget(*renderTarget).AddReadResource(*m_RDGParticlePosition);
    }

    // MPMSimulator implementation

    IFRIT_APIDECL MPMSimulator::MPMSimulator() : m_Data(new MPMSimulatorPrivateData())
    {
        m_Data->m_Config              = &m_Config;
        m_Data->m_RebuildGPUResources = true;
    }
    IFRIT_APIDECL MPMSimulator::~MPMSimulator()
    {
        delete m_Data;
        m_Data = nullptr;
    }
    IFRIT_APIDECL void MPMSimulator::SetConfig(const MPMSimulatorConfig& cfg)
    {
        m_Config         = cfg;
        m_Data->m_Config = &m_Config;
    }

    IFRIT_APIDECL void MPMSimulator::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        m_Data->RunSolverStep(builder, deltaTime);
    }

    IFRIT_APIDECL void MPMSimulator::Render(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        if (m_Config.m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
        {
            m_Data->ParticleRender2D(builder, renderTarget);
        }
        else if (m_Config.m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
        {
            m_Data->ParticleRender3D(builder, renderTarget);
        }
        else
        {
            iAssertion(false, "MPMSimulator: Invalid problem dimension specified for rendering.");
        }
    }

} // namespace Ifrit::Runtime::Siro