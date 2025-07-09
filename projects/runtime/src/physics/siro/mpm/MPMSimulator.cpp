#include "ifrit/runtime/physics/siro/mpm/MPMSimulator.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Siro.h"
#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit/core/math/linalg/LinalgOps.h"
#include <variant>

using namespace Ifrit::Math;
using namespace Ifrit::RHI;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime::Siro
{
    template <u32 Dimension> struct MPMSimulatorTypes;

    struct MPMEmissionInfo
    {
        std::variant<Vec<Vector2f>, Vec<Vector4f>> m_InitialParticleLocations;
        MPMParticleEmitArgs                        m_EmissionArgs;
    };

    struct MPMParticleMaterials
    {
        f32                      m_YoungsModulus = 100.0f;
        f32                      m_PoissonRatio  = 0.2f;
        MPMSimulatorParticleType m_ParticleType  = MPMSimulatorParticleType::Fluid;
    };

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
        static IF_CONSTEXPR u32                    kDefaultTGX = IfritShader::Siro::MPM::kMpmTGSizeX;

        MPMSimulatorConfig*                        m_Config                   = nullptr;
        bool                                       m_RebuildGPUResources      = true;
        bool                                       m_HasInitParticleLocations = false;

        std::variant<Vec<Vector2f>, Vec<Vector4f>> m_InitParticleLocations;
        Vec<MPMEmissionInfo>                       m_EmissionInfos;

        // Persistent data
        RhiBufferRef                               m_ParticleCount;
        RhiBufferRef                               m_ParticleEmitLocations;

        RhiBufferRef                               m_ParticlePosition;
        RhiBufferRef                               m_ParticleVelocity;
        RhiBufferRef                               m_ParticleMass;
        RhiBufferRef                               m_ParticleDeformGrad;
        RhiBufferRef                               m_ParticleDeformGradDet;
        RhiBufferRef                               m_ParticleVolume;
        RhiBufferRef                               m_ParticleApicB;
        RhiBufferRef                               m_ParticleIndex;
        RhiBufferRef                               m_ParticleDebug;
        RhiBufferRef                               m_ParticleStressContrib;
        RhiBufferRef                               m_ParticleMatProperty;
        RhiBufferRef                               m_ParticleLiquidDensity; // For PBMPM

        RhiBufferRef                               m_GridForce;
        RhiBufferRef                               m_GridVelocity;
        RhiBufferRef                               m_GridMass;

        RhiBufferRef                               m_GridAttribute;

        // RDG resources
        FGBufferNodeRef                            m_RDGParticleCount;
        FGBufferNodeRef                            m_RDGParticleEmitLocations;

        FGBufferNodeRef                            m_RDGParticlePosition;
        FGBufferNodeRef                            m_RDGParticleVelocity;
        FGBufferNodeRef                            m_RDGParticleMass;
        FGBufferNodeRef                            m_RDGParticleDeformGrad;
        FGBufferNodeRef                            m_RDGParticleDeformGradDet;
        FGBufferNodeRef                            m_RDGParticleVolume;
        FGBufferNodeRef                            m_RDGParticleApicB;
        FGBufferNodeRef                            m_RDGParticleDebug;
        FGBufferNodeRef                            m_RDGParticleStressContrib;
        FGBufferNodeRef                            m_RDGParticleMatProperty;
        FGBufferNodeRef                            m_RDGParticleLiquidDensity; // For PBMPM

        FGBufferNodeRef                            m_RDGGridForce;
        FGBufferNodeRef                            m_RDGGridVelocity;
        FGBufferNodeRef                            m_RDGGridMass;
        FGBufferNodeRef                            m_RDGGridAttribute;

        // Transient data
        FGBufferNodeRef                            m_RDGValidGridCounter;
        FGBufferNodeRef                            m_RDGValidGridList;

        // Init
        template <u32 Dimension> void              InitGPUResources(RhiBackend* RHI);
        void                                       PrepareInitialGPUData(RhiBackend* RHI);
        void                                       InitRDGResources(FrameGraphBuilder& builder);
        void                                       RunSolverStep(FrameGraphBuilder& builder, f32 dt);
        ShaderVariantDesc                          GetShader(const String& name, const Vec<String>& extra = {}) const;
        u32                                        GetNumGrids() const;
        template <u32 Dimension> void              HandleManualParticleEmit(FrameGraphBuilder& builder);

        // Solver steps
        void                                       ParticleInit(FrameGraphBuilder& builder);
        void ParticleEmit(FrameGraphBuilder& builder, const MPMParticleEmitArgs& args, u32 numParticles);
        void GridReset(FrameGraphBuilder& builder, bool firstFrame, bool firstIteration);
        void ParticleToGridTransfer(FrameGraphBuilder& builder, f32 dt, u32 last);
        void GridVelocityNormalize(FrameGraphBuilder& builder);
        void GridForceUpdate(FrameGraphBuilder& builder);
        void GridGravityApply(FrameGraphBuilder& builder);
        void GridVelocityUpdate(FrameGraphBuilder& builder, f32 dt, bool firstIteration);
        void GridToParticleTransfer(FrameGraphBuilder& builder, f32 dt);
        void ParticleAdvect(FrameGraphBuilder& builder, f32 dt);

        // Position-based MPM
        void PbMpmResolveConstraints(FrameGraphBuilder& builder, f32 dt);
        void PbMpmParticleIntegrate(FrameGraphBuilder& builder, f32 dt);

        // Visualizer
        void ParticleRender2D(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
        void ParticleRender3D(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
    };

    void MPMSimulatorPrivateData::ParticleEmit(
        FrameGraphBuilder& builder, const MPMParticleEmitArgs& args, u32 numParticles)
    {
        struct PushConst
        {
            f32 m_DefaultYoungs;
            f32 m_DefaultPossion;
            i32 m_DefaultMatType;

            i32 m_NumParticles;
            f32 m_Mass;
            f32 m_Density;
            u32 m_Grid;
            u32 m_ParticleLocationSrc;

            u32 m_ParticleLocation;
            u32 m_ParticleVelocity;
            u32 m_ParticleMass;
            u32 m_ParticleVolume;
            u32 m_ParticleB;
            u32 m_ParticleDeformationGrad;
            u32 m_ParticleDeformationGradDet;
            u32 m_ParticleMaterial;
            u32 m_ParticleLiquidDensity;
            u32 m_ParticleCounter;
        } pc;

        pc.m_DefaultYoungs  = args.m_YoungsModulus;
        pc.m_DefaultPossion = args.m_PoissonRatio;
        pc.m_DefaultMatType = static_cast<i32>(args.m_MaterialType);
        pc.m_NumParticles   = static_cast<i32>(numParticles);
        pc.m_Mass           = args.m_Mass;
        pc.m_Density        = args.m_Density;

        i32 tgX = DivRoundUp(numParticles, kDefaultTGX);

        AddComputePass<PushConst>(builder, "MPMSimulator.ParticleEmit",
            GetShader(Internal::kIntShaderTableSiro.MPMParticleEmitCS), Vector3i(tgX, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_Grid                       = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleLocationSrc        = ctx.m_FgDesc->GetUAV(*m_RDGParticleEmitLocations);
                pc.m_ParticleLocation           = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleVelocity           = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleMass               = ctx.m_FgDesc->GetUAV(*m_RDGParticleMass);
                pc.m_ParticleVolume             = ctx.m_FgDesc->GetUAV(*m_RDGParticleVolume);
                pc.m_ParticleB                  = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_ParticleDeformationGrad    = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleDeformationGradDet = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGradDet);
                pc.m_ParticleMaterial           = ctx.m_FgDesc->GetUAV(*m_RDGParticleMatProperty);
                pc.m_ParticleLiquidDensity      = ctx.m_FgDesc->GetUAV(*m_RDGParticleLiquidDensity);
                pc.m_ParticleCounter            = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGGridAttribute)
            .AddReadWriteResource(*m_RDGParticlePosition)
            .AddReadWriteResource(*m_RDGParticleVelocity)
            .AddReadWriteResource(*m_RDGParticleMass)
            .AddReadWriteResource(*m_RDGParticleDeformGrad)
            .AddReadWriteResource(*m_RDGParticleDeformGradDet)
            .AddReadWriteResource(*m_RDGParticleApicB)
            .AddReadWriteResource(*m_RDGParticleVolume)
            .AddReadWriteResource(*m_RDGParticleMatProperty)
            .AddReadWriteResource(*m_RDGParticleLiquidDensity)
            .AddReadWriteResource(*m_RDGParticleEmitLocations)
            .AddReadWriteResource(*m_RDGParticleCount);
    }

    void MPMSimulatorPrivateData::PbMpmParticleIntegrate(FrameGraphBuilder& builder, f32 dt)
    {
        struct PushConst
        {
            Vector4f m_Gravity;
            u32      m_ParticleCounterBuf;
            f32      m_DeltaTime;

            u32      m_Grid;
            u32      m_ParticleLocation;
            u32      m_ParticleVelocity; // !!! Particle Displacement Indeed !!!
            u32      m_ParticleDeformationGrad;
            u32      m_ParticleB;
            u32      m_ParticleMaterialSRV;
            u32      m_ParticleLiquidDensity;
            u32      m_ParticleDebug;

            f32      m_ViscoPlasticity;
        } pc;
        pc.m_Gravity         = Vector4f(m_Config->m_Gravity, 0.0f);
        pc.m_DeltaTime       = dt;
        pc.m_ViscoPlasticity = m_Config->m_DefaultViscoPlasticity;

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.PbMpmParticleIntegrate",
            GetShader(Internal::kIntShaderTableSiro.MPMPbMpmParticleIntegrateCS), *m_RDGParticleCount, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterBuf      = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                pc.m_Grid                    = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleLocation        = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleVelocity        = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleDeformationGrad = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleB               = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_ParticleMaterialSRV     = ctx.m_FgDesc->GetSRV(*m_RDGParticleMatProperty);
                pc.m_ParticleLiquidDensity   = ctx.m_FgDesc->GetUAV(*m_RDGParticleLiquidDensity);
                pc.m_ParticleDebug           = ctx.m_FgDesc->GetUAV(*m_RDGParticleDebug);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGGridAttribute)
            .AddReadWriteResource(*m_RDGParticlePosition)
            .AddReadWriteResource(*m_RDGParticleDebug)
            .AddReadWriteResource(*m_RDGParticleVelocity)
            .AddReadWriteResource(*m_RDGParticleDeformGrad)
            .AddReadWriteResource(*m_RDGParticleApicB)
            .AddReadWriteResource(*m_RDGParticleLiquidDensity)
            .AddReadResource(*m_RDGParticleMatProperty);
    }

    void MPMSimulatorPrivateData::PbMpmResolveConstraints(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            u32 m_ParticleCounterBuf;
            f32 m_DeltaTime;
            u32 m_ParticleVelocity;
            u32 m_ParticleB;
            u32 m_ParticleDeformationGrad;
            u32 m_ParticleMaterial;
            u32 m_Grid;
            u32 m_ParticleDebug;
            u32 m_ParticleLiquidDensity;

            // PBMPM
            f32 m_ElasticityInterpolationFactor;
            f32 m_ElasticityRelaxationFactor;
            f32 m_LiquidViscosity;
            f32 m_LiquidRelaxation;
            f32 m_ViscoPlasticity;
        } pc;

        pc.m_DeltaTime                     = deltaTime;
        pc.m_ElasticityInterpolationFactor = m_Config->m_PbMpmDefaultElasticityInterpolationFactor;
        pc.m_ElasticityRelaxationFactor    = m_Config->m_PbMpmDefaultElasticityRelaxationFactor;
        pc.m_LiquidViscosity               = m_Config->m_PbMpmDefaultLiquidViscosity;
        pc.m_LiquidRelaxation              = m_Config->m_PbMpmDefaultLiquidRelaxation;
        pc.m_ViscoPlasticity               = m_Config->m_DefaultViscoPlasticity;

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.PbMpmResolveConstraints",
            GetShader(Internal::kIntShaderTableSiro.MPMPbMpmResolveConstraintsCS), *m_RDGParticleCount, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterBuf      = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                pc.m_ParticleVelocity        = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleB               = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_ParticleDeformationGrad = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleMaterial        = ctx.m_FgDesc->GetUAV(*m_RDGParticleMatProperty);
                pc.m_ParticleDebug           = ctx.m_FgDesc->GetUAV(*m_RDGParticleDebug);
                pc.m_Grid                    = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleLiquidDensity   = ctx.m_FgDesc->GetUAV(*m_RDGParticleLiquidDensity);

                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGParticleVelocity)
            .AddReadWriteResource(*m_RDGParticleApicB)
            .AddReadWriteResource(*m_RDGParticleDeformGrad)
            .AddReadWriteResource(*m_RDGGridAttribute)
            .AddReadWriteResource(*m_RDGParticleDebug)
            .AddReadWriteResource(*m_RDGParticleLiquidDensity)
            .AddReadWriteResource(*m_RDGParticleMatProperty);
    }

    void MPMSimulatorPrivateData::ParticleInit(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_IgnoreParticlePosition;
            f32 m_DefaultYoungsModulus;
            f32 m_DefaultPoissonRatio;
            u32 m_DefaultMatType;

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
            u32 m_ParticleMatProperty;
            u32 m_ParticleLiquidDensity;
            u32 m_ParticleCount;
        } pc;
        pc.m_IgnoreParticlePosition = m_HasInitParticleLocations ? 1 : 0;
        pc.m_DefaultYoungsModulus   = m_Config->m_DefaultYoungsModulus;
        pc.m_DefaultPoissonRatio    = m_Config->m_DefaultPoissonRatio;
        pc.m_DefaultMatType         = static_cast<u32>(m_Config->m_DefaultParticleType);

        pc.m_NumParticles = m_Config->m_DefaultNumParticles;
        pc.m_Mass         = m_Config->m_DefaultMass;
        pc.m_Density      = m_Config->m_DefaultDensity;

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
                pc.m_ParticleMatProperty        = ctx.m_FgDesc->GetUAV(*m_RDGParticleMatProperty);
                pc.m_ParticleLiquidDensity      = ctx.m_FgDesc->GetUAV(*m_RDGParticleLiquidDensity);
                pc.m_ParticleCount              = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);

                SetRootConstant(pc, ctx);
            })
            .AddWriteResource(*m_RDGParticlePosition)
            .AddWriteResource(*m_RDGParticleVelocity)
            .AddWriteResource(*m_RDGParticleMass)
            .AddWriteResource(*m_RDGParticleVolume)
            .AddWriteResource(*m_RDGParticleApicB)
            .AddWriteResource(*m_RDGParticleDeformGrad)
            .AddWriteResource(*m_RDGParticleDeformGradDet)
            .AddWriteResource(*m_RDGParticleMatProperty)
            .AddReadResource(*m_RDGGridAttribute);
    }

    void MPMSimulatorPrivateData::GridReset(FrameGraphBuilder& builder, bool firstFrame, bool isFirstIteration)
    {
        struct PushConst
        {
            u32 m_FirstTime;
            u32 m_Grid;
            u32 m_ValidGridCounter;
            u32 m_ValidGridList;
        } pc;
        pc.m_FirstTime        = isFirstIteration ? 1 : 0;
        pc.m_ValidGridCounter = 0;
        pc.m_Grid             = 0;
        pc.m_ValidGridList    = 0;

        auto        numGrids = GetNumGrids();
        auto        tgX      = static_cast<i32>(DivRoundUp(numGrids, kDefaultTGX));
        Vec<String> extra;
        bool        isPbMpm = m_Config->m_Variant == MPMSimulatorVariant::PBMPM;
        if (isPbMpm)
        {
            extra.push_back("IFSHADER_MPM_PBMPM");
        }

        if (firstFrame || true)
        {
            extra.push_back("IFSHADER_MPM_GRIDRESET_INIT");
            AddComputePass<PushConst>(builder, "MPMSimulator.GridReset.Init",
                GetShader(Internal::kIntShaderTableSiro.MPMGridResetCS, extra), Vector3i(tgX, 1, 1), pc,
                [this](PushConst pc, const FrameGraphPassContext& ctx) {
                    pc.m_Grid             = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                    pc.m_ValidGridCounter = ctx.m_FgDesc->GetUAV(*m_RDGValidGridCounter);
                    pc.m_ValidGridList    = ctx.m_FgDesc->GetUAV(*m_RDGValidGridList);
                    SetRootConstant(pc, ctx);
                })
                .AddWriteResource(*m_RDGGridAttribute)
                .AddReadResource(*m_RDGValidGridCounter)
                .AddWriteResource(*m_RDGValidGridCounter)
                .AddReadResource(*m_RDGValidGridList);
        }
        else
        {
            AddIndirectComputePass<PushConst>(builder, "MPMSimulator.GridReset.Seq",
                GetShader(Internal::kIntShaderTableSiro.MPMGridResetCS, extra), *m_RDGValidGridCounter, sizeof(u32), pc,
                [this](PushConst pc, const FrameGraphPassContext& ctx) {
                    pc.m_Grid             = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                    pc.m_ValidGridCounter = ctx.m_FgDesc->GetUAV(*m_RDGValidGridCounter);
                    pc.m_ValidGridList    = ctx.m_FgDesc->GetUAV(*m_RDGValidGridList);
                    SetRootConstant(pc, ctx);
                })
                .AddWriteResource(*m_RDGGridAttribute)
                .AddReadWriteResource(*m_RDGValidGridCounter)
                .AddReadResource(*m_RDGValidGridList);
        }
    }

    void MPMSimulatorPrivateData::ParticleToGridTransfer(FrameGraphBuilder& builder, f32 deltaTime, u32 firstOrLastRun)
    {
        struct PushConst
        {
            u32 m_ParticleCounterBuf;
            f32 m_DeltaTime;
            u32 m_ParticleVelocity;
            u32 m_ParticleLocation;
            u32 m_ParticleMass;
            u32 m_ParticleB;
            u32 m_Grid;
            u32 m_ParticleDeformGrad;
            u32 m_ParticleDeformGradDet;
            u32 m_ParticleDebug;
            u32 m_ParticleStressContrib;
            u32 m_ParticleMatProperty;
            u32 m_IsFirstOrLastRun;
        } pc;

        pc.m_DeltaTime             = deltaTime;
        pc.m_ParticleVelocity      = 0;
        pc.m_ParticleLocation      = 0;
        pc.m_ParticleMass          = 0;
        pc.m_ParticleB             = 0;
        pc.m_Grid                  = 0;
        pc.m_ParticleDeformGrad    = 0;
        pc.m_ParticleDeformGradDet = 0;
        pc.m_ParticleDebug         = 0;
        pc.m_ParticleStressContrib = 0;
        pc.m_ParticleMatProperty   = 0;
        pc.m_IsFirstOrLastRun      = firstOrLastRun;

        Vec<String> extra;
        auto        isPbMpm = m_Config->m_Variant == MPMSimulatorVariant::PBMPM;
        if (isPbMpm)
        {
            extra.push_back("IFSHADER_MPM_PBMPM");
        }

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.ParticleToGridTransfer",
            GetShader(Internal::kIntShaderTableSiro.MPMP2GCS, extra), *m_RDGParticleCount, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterBuf    = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                pc.m_ParticleVelocity      = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                pc.m_ParticleLocation      = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleMass          = ctx.m_FgDesc->GetUAV(*m_RDGParticleMass);
                pc.m_ParticleB             = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_Grid                  = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleDeformGrad    = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleDeformGradDet = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGradDet);
                pc.m_ParticleDebug         = ctx.m_FgDesc->GetUAV(*m_RDGParticleDebug);
                pc.m_ParticleStressContrib = ctx.m_FgDesc->GetUAV(*m_RDGParticleStressContrib);
                pc.m_ParticleMatProperty   = ctx.m_FgDesc->GetUAV(*m_RDGParticleMatProperty);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleCount)
            .AddReadResource(*m_RDGParticleVelocity)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleMass)
            .AddReadResource(*m_RDGParticleApicB)
            .AddWriteResource(*m_RDGGridAttribute)
            .AddWriteResource(*m_RDGGridVelocity)
            .AddWriteResource(*m_RDGGridMass)
            .AddWriteResource(*m_RDGParticleDebug)
            .AddReadWriteResource(*m_RDGParticleDeformGradDet)
            .AddReadWriteResource(*m_RDGParticleStressContrib)
            .AddReadResource(*m_RDGParticleMatProperty)
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
            u32 m_ParticleCounterBuf;
            u32 m_Grid;
            u32 m_ParticleLocation;
            u32 m_ParticleVolume;
            u32 m_ParticleDeformationGrad;
            u32 m_ParticleDeformationGradDet;
            u32 m_ParticleStressContrib;
            u32 m_ParticleMaterial;
        } pc;
        auto Mu     = m_Config->m_DefaultYoungsModulus;
        auto Lambda = m_Config->m_DefaultPoissonRatio;

        pc.m_Grid                       = 0;
        pc.m_ParticleLocation           = 0;
        pc.m_ParticleVolume             = 0;
        pc.m_ParticleDeformationGrad    = 0;
        pc.m_ParticleDeformationGradDet = 0;
        pc.m_ParticleStressContrib      = 0;
        pc.m_ParticleMaterial           = 0;

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.GridForceUpdate",
            GetShader(Internal::kIntShaderTableSiro.MPMGridForceUpdateCS), *m_RDGParticleCount, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterBuf         = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                pc.m_Grid                       = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleLocation           = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleVolume             = ctx.m_FgDesc->GetUAV(*m_RDGParticleVolume);
                pc.m_ParticleDeformationGrad    = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleDeformationGradDet = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGradDet);
                pc.m_ParticleStressContrib      = ctx.m_FgDesc->GetUAV(*m_RDGParticleStressContrib);
                pc.m_ParticleMaterial           = ctx.m_FgDesc->GetUAV(*m_RDGParticleMatProperty);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleCount)
            .AddReadResource(*m_RDGGridAttribute)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleVolume)
            .AddReadResource(*m_RDGParticleDeformGrad)
            .AddReadResource(*m_RDGParticleStressContrib)
            .AddReadWriteResource(*m_RDGParticleDeformGradDet)
            .AddReadResource(*m_RDGParticleMatProperty)
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

    void MPMSimulatorPrivateData::GridVelocityUpdate(FrameGraphBuilder& builder, f32 deltaTime, bool firstIteration)
    {
        struct PushConst
        {
            f32 m_DeltaTime;
            u32 m_Grid;
            u32 m_ValidGridCounter;
            u32 m_ValidGridList;
            u32 m_RequireNormalizeVelocity;
        } pc;

        pc.m_DeltaTime        = deltaTime;
        pc.m_Grid             = 0;
        pc.m_ValidGridCounter = 0;
        pc.m_ValidGridList    = 0;

        Vec<String> extra;
        auto        isPbMpm           = m_Config->m_Variant == MPMSimulatorVariant::PBMPM;
        pc.m_RequireNormalizeVelocity = (firstIteration && isPbMpm) ? 1 : 0;

        if (isPbMpm)
        {
            extra.push_back("IFSHADER_MPM_PBMPM");
        }

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.GridVelocityUpdate",
            GetShader(Internal::kIntShaderTableSiro.MPMGridVelocityUpdateCS, extra), *m_RDGValidGridCounter,
            sizeof(u32), pc,
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
            u32 m_ParticleCounterBuf;
            f32 m_DeltaTime;
            u32 m_ParticleLocation;
            u32 m_ParticleVelocity;
        } pc;
        pc.m_DeltaTime        = deltaTime;
        pc.m_ParticleLocation = 0;
        pc.m_ParticleVelocity = 0;

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.ParticleAdvect",
            GetShader(Internal::kIntShaderTableSiro.MPMParticleAdvectionCS), *m_RDGParticleCount, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterBuf = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                pc.m_ParticleLocation   = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleVelocity   = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleCount)
            .AddReadWriteResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleVelocity);
    }

    void MPMSimulatorPrivateData::PrepareInitialGPUData(RhiBackend* RHI)
    {
        auto          numParticles        = m_Config->m_DefaultNumParticles;
        auto          maxParticles        = m_Config->m_MaxParticles;
        Array<u32, 4> particleDataSection = { numParticles, 0, 1, 1 };

        Vec<u32>      particleIndexData(maxParticles);
        for (u32 i = 0; i < maxParticles; ++i)
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
        auto stagedPosition = RHI->CreateStagedSingleBuffer(m_ParticlePosition.get());
        auto stagedCounter  = RHI->CreateStagedSingleBuffer(m_ParticleCount.get());
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            stagedIndex->CmdCopyToDevice(cmd, particleIndexData.data(), particleIndexData.size() * sizeof(u32), 0);
            stagedGridAttr->CmdCopyToDevice(cmd, &gridAttr, sizeof(MPMSimulatorGridAttribute), 0);
            stagedCounter->CmdCopyToDevice(
                cmd, particleDataSection.data(), particleDataSection.size() * sizeof(u32), 0);

            if (m_HasInitParticleLocations)
            {
                if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
                {
                    if (std::holds_alternative<Vec<Vector2f>>(m_InitParticleLocations))
                    {
                        auto& initLocs = std::get<Vec<Vector2f>>(m_InitParticleLocations);
                        stagedPosition->CmdCopyToDevice(cmd, initLocs.data(), initLocs.size() * sizeof(Vector2f), 0);
                    }
                    else
                    {
                        iAssertion(false,
                            "MPMSimulator: Initial particle locations must be of type Vec<Vector2f> "
                            "for 2D simulations.");
                    }
                }
                else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
                {
                    if (std::holds_alternative<Vec<Vector4f>>(m_InitParticleLocations))
                    {
                        auto& initLocs = std::get<Vec<Vector4f>>(m_InitParticleLocations);
                        stagedPosition->CmdCopyToDevice(cmd, initLocs.data(), initLocs.size() * sizeof(Vector4f), 0);
                    }
                    else
                    {
                        iAssertion(false,
                            "MPMSimulator: Initial particle locations must be of type Vec<Vector4f> "
                            "for 3D simulations.");
                    }
                }
            }
        });
    }

    void MPMSimulatorPrivateData::GridToParticleTransfer(FrameGraphBuilder& builder, f32 deltaTime)
    {
        struct PushConst
        {
            u32 m_ParticleCounterBuf;
            f32 m_DeltaTime;
            u32 m_Grid;
            u32 m_ParticleDeformationGrad;
            u32 m_ParticleLocation;
            u32 m_ParticleB;
            u32 m_ParticleVelocity;
        } pc;

        pc.m_DeltaTime               = deltaTime;
        pc.m_Grid                    = 0;
        pc.m_ParticleDeformationGrad = 0;
        pc.m_ParticleLocation        = 0;
        pc.m_ParticleB               = 0;
        pc.m_ParticleVelocity        = 0;

        Vec<String> extra;
        auto        isPbMpm = m_Config->m_Variant == MPMSimulatorVariant::PBMPM;
        if (isPbMpm)
        {
            extra.push_back("IFSHADER_MPM_PBMPM");
        }

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.GridToParticleTransfer",
            GetShader(Internal::kIntShaderTableSiro.MPMG2PCS, extra), *m_RDGParticleCount, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounterBuf      = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                pc.m_Grid                    = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_ParticleDeformationGrad = ctx.m_FgDesc->GetUAV(*m_RDGParticleDeformGrad);
                pc.m_ParticleLocation        = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
                pc.m_ParticleB               = ctx.m_FgDesc->GetUAV(*m_RDGParticleApicB);
                pc.m_ParticleVelocity        = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleCount)
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

    template <u32 Dimension> void MPMSimulatorPrivateData::HandleManualParticleEmit(FrameGraphBuilder& builder)
    {
        auto rhi    = builder.GetRhi();
        using EleTp = MPMSimulatorTypes<Dimension>::FSpatialVectorAligned;
        using VecTp = Vec<EleTp>;
        for (auto& emitRequest : m_EmissionInfos)
        {
            iAssertion(m_EmissionInfos.size() == 1,
                "MPMSimulator: Manual particle emission is only supported for a single emission request at a time.");
            if (std::holds_alternative<VecTp>(emitRequest.m_InitialParticleLocations))
            {
                auto  stagedLocation = rhi->CreateStagedSingleBuffer(m_ParticleEmitLocations.get());
                auto  tq             = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
                auto& locations      = std::get<VecTp>(emitRequest.m_InitialParticleLocations);
                tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                    stagedLocation->CmdCopyToDevice(cmd, locations.data(), locations.size() * sizeof(EleTp), 0);
                });
                iDebug("Emit start!");
                ParticleEmit(builder, emitRequest.m_EmissionArgs, locations.size());
                iDebug("Emit end!");
                m_Config->m_DefaultNumParticles += static_cast<u32>(locations.size());
            }
            else
            {
                iAssertion(false, "MPMSimulator: particle dimension mismatches.");
            }
        }
        m_EmissionInfos.clear();
    }

    template <u32 Dimension> void MPMSimulatorPrivateData::InitGPUResources(RhiBackend* RHI)
    {
        using MTypes = MPMSimulatorTypes<Dimension>;
        iAssertion(m_Config, "MPMSimulator: Config must be set before initializing GPU resources.");
        static_assert(Dimension == 2 || Dimension == 3, "MPMSimulator: Invalid dimension specified.");

        auto numParticles = m_Config->m_MaxParticles;
        auto numGrids     = GetNumGrids();

        auto particleCountSz = sizeof(u32) * 4;

        auto particlePosSz           = numParticles * MTypes::kFSpatialVectorAlignedSize;
        auto particleVelSz           = numParticles * MTypes::kFSpatialVectorAlignedSize;
        auto particleMassSz          = numParticles * MTypes::kFScalarSize;
        auto particleDeformGradSz    = numParticles * MTypes::kFSpatialTransformAlignedSize;
        auto particleJSz             = numParticles * MTypes::kFScalarSize;
        auto particleVolSz           = numParticles * MTypes::kFScalarSize;
        auto particleApicBSz         = numParticles * MTypes::kFSpatialTransformAlignedSize;
        auto particleIndexSz         = numParticles * sizeof(u32);
        auto particleDebugSz         = numParticles * 64;
        auto particleStressContribSz = numParticles * MTypes::kFSpatialTransformAlignedSize;
        auto particleMatPropertySz   = numParticles * sizeof(MPMParticleMaterials);
        auto particleLiquidSz        = numParticles * sizeof(f32);

        auto gridForceSz = numGrids * MTypes::kFSpatialVectorAlignedSize;
        auto gridVelSz   = numGrids * MTypes::kFSpatialVectorAlignedSize;
        auto gridMassSz  = numGrids * MTypes::kFScalarSize;
        auto gridAttrSz  = sizeof(MPMSimulatorGridAttribute);

        auto defaultUsage  = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
        auto indexUsage    = defaultUsage | RhiBufferUsage::RhiBufferUsage_Index;
        auto indirectUsage = defaultUsage | RhiBufferUsage::RhiBufferUsage_Indirect;

        m_ParticleCount    = RHI->CreateBufferDevice("MPM_ParticleCount", particleCountSz, indirectUsage, true);
        m_ParticlePosition = RHI->CreateBufferDevice("MPM_ParticlePosition", particlePosSz, defaultUsage, true);
        m_ParticleEmitLocations =
            RHI->CreateBufferDevice("MPM_ParticleEmitLocations", particlePosSz, defaultUsage, true);
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
        m_ParticleStressContrib =
            RHI->CreateBufferDevice("MPM_ParticleStressContrib", particleStressContribSz, defaultUsage, true);
        m_ParticleMatProperty =
            RHI->CreateBufferDevice("MPM_ParticleMaterialProperty", particleMatPropertySz, defaultUsage, true);
        m_ParticleLiquidDensity =
            RHI->CreateBufferDevice("MPM_ParticleLiquiddDensity", particleLiquidSz, defaultUsage, true);

        m_GridForce     = RHI->CreateBufferDevice("MPM_GridForce", gridForceSz, defaultUsage, true);
        m_GridVelocity  = RHI->CreateBufferDevice("MPM_GridVelocity", gridVelSz, defaultUsage, true);
        m_GridMass      = RHI->CreateBufferDevice("MPM_GridMass", gridMassSz, defaultUsage, true);
        m_GridAttribute = RHI->CreateBufferDevice("MPM_GridAttribute", gridAttrSz, defaultUsage, true);

        PrepareInitialGPUData(RHI);
    }

    void MPMSimulatorPrivateData::InitRDGResources(FrameGraphBuilder& builder)
    {
        // Persistent
        m_RDGParticleCount         = &builder.ImportBuffer("MPM_ParticleCount", m_ParticleCount.get());
        m_RDGParticlePosition      = &builder.ImportBuffer("MPM_ParticlePosition", m_ParticlePosition.get());
        m_RDGParticleEmitLocations = &builder.ImportBuffer("MPM_ParticleEmitLocations", m_ParticleEmitLocations.get());
        m_RDGParticleVelocity      = &builder.ImportBuffer("MPM_ParticleVelocity", m_ParticleVelocity.get());
        m_RDGParticleMass          = &builder.ImportBuffer("MPM_ParticleMass", m_ParticleMass.get());
        m_RDGParticleDeformGrad    = &builder.ImportBuffer("MPM_ParticleDeformGradient", m_ParticleDeformGrad.get());
        m_RDGParticleDeformGradDet =
            &builder.ImportBuffer("MPM_ParticleDeformGradientDeterminant", m_ParticleDeformGradDet.get());
        m_RDGParticleVolume        = &builder.ImportBuffer("MPM_ParticleVolume", m_ParticleVolume.get());
        m_RDGParticleApicB         = &builder.ImportBuffer("MPM_ParticleApicB", m_ParticleApicB.get());
        m_RDGParticleDebug         = &builder.ImportBuffer("MPM_ParticleDebug", m_ParticleDebug.get());
        m_RDGParticleStressContrib = &builder.ImportBuffer("MPM_ParticleStressContrib", m_ParticleStressContrib.get());
        m_RDGParticleMatProperty   = &builder.ImportBuffer("MPM_ParticleMaterialProperty", m_ParticleMatProperty.get());
        m_RDGParticleLiquidDensity = &builder.ImportBuffer("MPM_ParticleLiquidDensity", m_ParticleLiquidDensity.get());

        m_RDGGridForce     = &builder.ImportBuffer("MPM_GridForce", m_GridForce.get());
        m_RDGGridVelocity  = &builder.ImportBuffer("MPM_GridVelocity", m_GridVelocity.get());
        m_RDGGridMass      = &builder.ImportBuffer("MPM_GridMass", m_GridMass.get());
        m_RDGGridAttribute = &builder.ImportBuffer("MPM_GridAttribute", m_GridAttribute.get());

        // Transient
        // TODO: make imported !!!
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
        bool isPbMpm    = (m_Config->m_Variant == MPMSimulatorVariant::PBMPM);
        if (m_RebuildGPUResources) IF_UNLIKELY
        {
            m_RebuildGPUResources = false;
            if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
                InitGPUResources<2>(rhi);
            else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
                InitGPUResources<3>(rhi);
            else IF_UNLIKELY
                iAssertion(false, "MPMSimulator: Invalid problem dimension specified.");
        }
        InitRDGResources(builder);
        if (isFirstRun) IF_UNLIKELY
        {
            ParticleInit(builder);
        }

        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
            HandleManualParticleEmit<3>(builder);
        else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
            HandleManualParticleEmit<3>(builder);

        if (isPbMpm)
        {
            for (auto i = 0; i < m_Config->m_Substeps; ++i)
            {
                {
                    IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.PbMpmSubstep");

                    for (auto j = 0; j < m_Config->m_PbMpmIterations; ++j)
                    {
                        bool isLastIteration  = (j == m_Config->m_PbMpmIterations - 1);
                        bool isFirstIteration = (j == 0);
                        u32  firstOrLastRun   = 0;
                        firstOrLastRun |= (isFirstIteration) ? 1 : 0;
                        firstOrLastRun |= (isLastIteration) ? 2 : 0;
                        {
                            IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.PbMpmIteration");
                            GridReset(builder, isFirstRun, isFirstIteration);
                            PbMpmResolveConstraints(builder, deltaTime);
                            ParticleToGridTransfer(builder, deltaTime, firstOrLastRun);
                            if (isFirstIteration)
                            {
                                GridVelocityNormalize(builder);
                            }
                            GridVelocityUpdate(builder, deltaTime, isFirstIteration);
                            GridToParticleTransfer(builder, deltaTime);
                        }
                        isFirstRun = false;
                    }
                    PbMpmParticleIntegrate(builder, deltaTime);
                }
            }
        }
        else
        {
            for (auto i = 0; i < m_Config->m_Substeps; ++i)
            {
                {
                    IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.MpmSubstep");
                    GridReset(builder, isFirstRun, true);
                    ParticleToGridTransfer(builder, deltaTime, false);
                    GridVelocityNormalize(builder);
                    GridForceUpdate(builder);
                    GridGravityApply(builder);
                    GridVelocityUpdate(builder, deltaTime, false);
                    GridToParticleTransfer(builder, deltaTime);
                    ParticleAdvect(builder, deltaTime);
                }
                isFirstRun = false;
            }
        }
    }

    ShaderVariantDesc MPMSimulatorPrivateData::GetShader(const String& name, const Vec<String>& extra) const
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
        for (const auto& e : extra)
        {
            shaderVariants.push_back(e);
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
        // Sleep(500);
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

    template <u32 Dimension IF_REQUIRES(Dimension == 2 || Dimension == 3)>
    void MPMSimulator::SetInitParticleLocations(const Vec<TGenericVector<f32, Dimension>>& locations)
    {
        IF_CONSTEXPR auto TAlignedDim = Dimension + (Dimension == 3 ? 1 : 0);
        using TAlignedVec             = TGenericVector<f32, TAlignedDim>;

        iAssertion(locations.size() > 0, "MPMSimulator: No particles found");

        Vec<TAlignedVec> alignedLocations(locations.size());
        for (u32 i = 0; i < locations.size(); ++i)
        {
            if IF_CONSTEXPR (Dimension == 2)
            {
                alignedLocations[i] = TAlignedVec(locations[i].x, locations[i].y);
            }
            else if IF_CONSTEXPR (Dimension == 3)
            {
                alignedLocations[i] = TAlignedVec(locations[i].x, locations[i].y, locations[i].z, 1.0f);
            }
        }
        m_Data->m_RebuildGPUResources           = true;
        m_Data->m_HasInitParticleLocations      = true;
        m_Data->m_Config->m_DefaultNumParticles = static_cast<u32>(locations.size());
        m_Data->m_InitParticleLocations         = std::move(alignedLocations);
    }

    template IFRIT_APIDECL void MPMSimulator::SetInitParticleLocations<2>(const Vec<TGenericVector<f32, 2>>& locations);
    template IFRIT_APIDECL void MPMSimulator::SetInitParticleLocations<3>(const Vec<TGenericVector<f32, 3>>& locations);

    template <u32 Dimension IF_REQUIRES(Dimension == 2 || Dimension == 3)>
    void MPMSimulator::EmitParticles(
        const Vec<TGenericVector<f32, Dimension>>& locations, const MPMParticleEmitArgs& args)

    {
        IF_CONSTEXPR auto TAlignedDim = Dimension + (Dimension == 3 ? 1 : 0);
        using TAlignedVec             = TGenericVector<f32, TAlignedDim>;
        Vec<TAlignedVec> alignedLocations(locations.size());
        for (u32 i = 0; i < locations.size(); ++i)

        {
            if IF_CONSTEXPR (Dimension == 2)
            {
                alignedLocations[i] = TAlignedVec(locations[i].x, locations[i].y);
            }
            else if IF_CONSTEXPR (Dimension == 3)
            {
                alignedLocations[i] = TAlignedVec(locations[i].x, locations[i].y, locations[i].z, 1.0f);
            }
        }
        MPMEmissionInfo emitInfo;
        emitInfo.m_InitialParticleLocations = alignedLocations;
        emitInfo.m_EmissionArgs             = args;
        m_Data->m_EmissionInfos.push_back(emitInfo);
    }
    template IFRIT_APIDECL void MPMSimulator::SetInitParticleLocations<2>(const Vec<TGenericVector<f32, 2>>& locations);
    template IFRIT_APIDECL void MPMSimulator::SetInitParticleLocations<3>(const Vec<TGenericVector<f32, 3>>& locations);
    template IFRIT_APIDECL void MPMSimulator::EmitParticles<2>(
        const Vec<TGenericVector<f32, 2>>& locations, const MPMParticleEmitArgs& args);
    template IFRIT_APIDECL void MPMSimulator::EmitParticles<3>(
        const Vec<TGenericVector<f32, 3>>& locations, const MPMParticleEmitArgs& args);

} // namespace Ifrit::Runtime::Siro