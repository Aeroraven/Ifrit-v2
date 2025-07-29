#include "ifrit/runtime/physics/artemis/mpm/MPMSimulator.h"
#include "ifrit/runtime/physics/internal/InternalShaderRegistry.Artemis.h"
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"
#include "ifrit/core/math/linalg/LinalgOps.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMParticleEmitter.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMParticleData.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMParticleContainer.h"
#include <variant>

#include "ifrit/runtime/physics/artemis/ArtemisSceneData.h"
#include "ifrit.internal/runtime/physics/artemis/InternalConst.h"
#include "ifrit.shader.neo/Artemis/Rigid/Rigid.Common.hlsli"
#include "ifrit.shader.neo/Shared/Artemis/MPMRigidCoupling.Shared.h"

using namespace Ifrit::Math;
using namespace Ifrit::RHI;
using namespace Ifrit::Runtime::FrameGraphUtils;

namespace Ifrit::Runtime::Artemis
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
        u32                                        m_FrameId            = 0;
        f32                                        m_ParticleRenderSize = 1.0f;
        static IF_CONSTEXPR u32                    kDefaultTGX          = IfritShader::Artemis::MPM::kMpmTGSizeX;

        MPMSimulatorConfig*                        m_Config                   = nullptr;
        bool                                       m_RebuildGPUResources      = true;
        bool                                       m_HasInitParticleLocations = false;
        bool                                       m_HasGlobalDrain           = false;
        bool                                       m_ShouldIntegrateRigids    = false;

        std::variant<Vec<Vector2f>, Vec<Vector4f>> m_InitParticleLocations;
        Vec<MPMEmissionInfo>                       m_EmissionInfos;
        RHI::RhiTexture*                           m_DebugRenderTarget = nullptr;

        // Persistent data
        RhiBufferRef                               m_ParticleEmitLocations;
        MPMGpuParticleBufferCollection*            m_ParticleData = nullptr;

        RhiBufferRef                               m_GridForce;
        RhiBufferRef                               m_GridVelocity;
        RhiBufferRef                               m_GridMass;

        RhiBufferRef                               m_GridAttribute;
        RhiBufferRef                               m_RenderParticleIndDrawBuffer;

        RhiBufferRef                               m_RigidContactCounter;
        RhiBufferRef                               m_RigidContactList;
        RhiBufferRef                               m_RigidBoundaryContactCounter;
        RhiBufferRef                               m_RigidBoundaryContactList;

        // RDG resources
        FGBufferNodeRef                            m_RDGParticleCount;
        FGBufferNodeRef                            m_RDGParticleEmitLocations;
        FGBufferNodeRef                            m_RDGParticleColor;

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

        FGBufferNodeRef                            m_RDGRenderParticleIndDrawBuffer;

        FGTextureNodeRef                           m_RDGRenderTarget;

        FGBufferNodeRef                            m_RDGRigidContactCounter;
        FGBufferNodeRef                            m_RDGRigidContactList;
        FGBufferNodeRef                            m_RDGRigidBoundaryContactCounter;
        FGBufferNodeRef                            m_RDGRigidBoundaryContactList;
        FGBufferNodeRef                            m_RDGRigidColliders;
        FGBufferNodeRef                            m_RDGRigidDynamics;

        // Transient data
        FGBufferNodeRef                            m_RDGValidGridCounter;
        FGBufferNodeRef                            m_RDGValidGridList;

        // Active scene data
        ArtemisSceneData*                          m_SceneData = nullptr;

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
        void ParticleDrainAll(FrameGraphBuilder& builder);
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

        // Position-based MPM Rigid Coupling
        void PbMpmRigidResetContactCounter(FrameGraphBuilder& builder);
        void PbMpmRigidCollectCollisionPairs(FrameGraphBuilder& builder);
        void PbMpmRigidIntegrate(FrameGraphBuilder& builder, f32 dt);
        void PbMpmRigidSyncTransform(FrameGraphBuilder& builder);
        void PbMpmRigidLoadTransform(FrameGraphBuilder& builder);
        void PbMpmRigidContactConstraintResolve(FrameGraphBuilder& builder);
        void PbMpmRigidCollectBoundaryContactPairs(FrameGraphBuilder& builder);
        void PbMpmRigidBoundaryConstraintResolve(FrameGraphBuilder& builder);

        // Visualizer
        void ParticleRender2D(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
        void ParticleRender3D(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
    };

    void MPMSimulatorPrivateData::PbMpmRigidCollectBoundaryContactPairs(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32             m_NumRigids;
            RHI::RhiSRVDesc m_RigidColliders;
            RHI::RhiSRVDesc m_RigidDynamics;
            RHI::RhiUAVDesc m_MpmGrid;
            RHI::RhiUAVDesc m_CollisionPairs;
            RHI::RhiUAVDesc m_CollisionPairCounter;
        } pc{};

        pc.m_NumRigids = m_SceneData->m_NumGpuColliders;

        int      tgX = Math::DivRoundUp(m_SceneData->m_NumGpuColliders, IfritShader::Artemis::Rigid::kRigidTGSizeX);
        Vector3i workGroup = Vector3i(tgX, 1, 1);

        AddComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidCollectBoundaryContactPairs",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMRigidCollectBoundaryContactCS), workGroup, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_RigidColliders       = ctx.m_FgDesc->GetSRV(*m_RDGRigidColliders);
                pc.m_RigidDynamics        = ctx.m_FgDesc->GetSRV(*m_RDGRigidDynamics);
                pc.m_MpmGrid              = ctx.m_FgDesc->GetUAV(*m_RDGGridAttribute);
                pc.m_CollisionPairs       = ctx.m_FgDesc->GetUAV(*m_RDGRigidBoundaryContactList);
                pc.m_CollisionPairCounter = ctx.m_FgDesc->GetUAV(*m_RDGRigidBoundaryContactCounter);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGRigidColliders)
            .AddReadResource(*m_RDGRigidDynamics)
            .AddReadWriteResource(*m_RDGGridAttribute)
            .AddReadWriteResource(*m_RDGRigidBoundaryContactList)
            .AddReadWriteResource(*m_RDGRigidBoundaryContactCounter);
    }

    void MPMSimulatorPrivateData::PbMpmRigidBoundaryConstraintResolve(FrameGraphBuilder& builder)
    {

        struct PushConst
        {
            RHI::RhiSRVDesc m_NumConstraints;
            RHI::RhiUAVDesc m_CollisionPairs;
            RHI::RhiSRVDesc m_RigidColliders;
            RHI::RhiUAVDesc m_RigidDynamics;
        } pc{};

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidBoundaryConstraintResolve",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMRigidBoundaryConstraintResolveCS),
            *m_RDGRigidBoundaryContactCounter, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_NumConstraints = ctx.m_FgDesc->GetSRV(*m_RDGRigidBoundaryContactCounter);
                pc.m_CollisionPairs = ctx.m_FgDesc->GetUAV(*m_RDGRigidBoundaryContactList);
                pc.m_RigidColliders = ctx.m_FgDesc->GetSRV(*m_RDGRigidColliders);
                pc.m_RigidDynamics  = ctx.m_FgDesc->GetUAV(*m_RDGRigidDynamics);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGRigidBoundaryContactCounter)
            .AddReadResource(*m_RDGRigidBoundaryContactList)
            .AddReadResource(*m_RDGRigidColliders)
            .AddReadWriteResource(*m_RDGRigidDynamics);
    }

    void MPMSimulatorPrivateData::PbMpmRigidContactConstraintResolve(FrameGraphBuilder& builder)
    {

        struct PushConst
        {
            RHI::RhiSRVDesc m_NumConstraints;
            RHI::RhiUAVDesc m_CollisionPairs;
            RHI::RhiSRVDesc m_ParticleLocations;
            RHI::RhiSRVDesc m_RigidColliders;
            RHI::RhiUAVDesc m_RigidDynamics;

            RHI::RhiSRVDesc m_ParticleMass;
            RHI::RhiUAVDesc m_ParticleDisplacements;
        } pc{};

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidContactConstraintResolve",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMRigidContactConstraintResolveCS),
            *m_RDGRigidContactCounter, sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_NumConstraints        = ctx.m_FgDesc->GetSRV(*m_RDGRigidContactCounter);
                pc.m_CollisionPairs        = ctx.m_FgDesc->GetUAV(*m_RDGRigidContactList);
                pc.m_ParticleLocations     = ctx.m_FgDesc->GetSRV(*m_RDGParticlePosition);
                pc.m_RigidColliders        = ctx.m_FgDesc->GetSRV(*m_RDGRigidColliders);
                pc.m_RigidDynamics         = ctx.m_FgDesc->GetUAV(*m_RDGRigidDynamics);
                pc.m_ParticleMass          = ctx.m_FgDesc->GetSRV(*m_RDGParticleMass);
                pc.m_ParticleDisplacements = ctx.m_FgDesc->GetUAV(*m_RDGParticleVelocity);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGRigidContactCounter)
            .AddReadResource(*m_RDGRigidContactList)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleMass)
            .AddReadWriteResource(*m_RDGParticleVelocity)
            .AddReadResource(*m_RDGRigidColliders)
            .AddReadWriteResource(*m_RDGRigidDynamics);
    }

    void MPMSimulatorPrivateData::PbMpmRigidLoadTransform(FrameGraphBuilder& builder)
    {
        // shader from rigid simulator
        struct PushConst
        {
            u32             m_NumRigidBodies;
            RHI::RhiSRVDesc m_IndirectCounter;
            RHI::RhiSRVDesc m_ColliderEntries;
            RHI::RhiUAVDesc m_ColliderDynamics;
        } pc{};
        pc.m_NumRigidBodies  = m_SceneData->m_NumGpuColliders;
        pc.m_IndirectCounter = ~0u;
        int      numTGX = Math::DivRoundUp(m_SceneData->m_NumGpuColliders, IfritShader::Artemis::Rigid::kRigidTGSizeX);
        Vector3i workGroup = Vector3i(numTGX, 1, 1);

        // IF_LOG_DEBUG("MPMSimulator", "Numrigids: {}", pc.m_NumRigidBodies);
        Vec<String> extra;
        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
        {
            extra.push_back("IFSHADER_RIGID_DYNAMICS_3D");
        }

        AddComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidLoadTransform",
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.RigidLoadTransformCS, extra), workGroup, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ColliderEntries  = ctx.m_FgDesc->GetSRV(*m_RDGRigidColliders);
                pc.m_ColliderDynamics = ctx.m_FgDesc->GetUAV(*m_RDGRigidDynamics);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGRigidColliders)
            .AddReadWriteResource(*m_RDGRigidDynamics);
    }

    void MPMSimulatorPrivateData::PbMpmRigidSyncTransform(FrameGraphBuilder& builder)
    {
        // shader from rigid simulator
        struct PushConst
        {
            u32             m_NumRigidBodies;
            RHI::RhiSRVDesc m_IndirectCounter;
            RHI::RhiSRVDesc m_ColliderEntries;
            RHI::RhiUAVDesc m_ColliderDynamics;
        } pc{};
        pc.m_NumRigidBodies  = m_SceneData->m_NumGpuColliders;
        pc.m_IndirectCounter = ~0u;
        int      numTGX = Math::DivRoundUp(m_SceneData->m_NumGpuColliders, IfritShader::Artemis::Rigid::kRigidTGSizeX);
        Vector3i workGroup = Vector3i(numTGX, 1, 1);

        Vec<String> extra;
        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
        {
            extra.push_back("IFSHADER_RIGID_DYNAMICS_3D");
        }

        AddComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidSyncTransform",
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.RigidSyncTransformCS, extra), workGroup, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ColliderEntries  = ctx.m_FgDesc->GetSRV(*m_RDGRigidColliders);
                pc.m_ColliderDynamics = ctx.m_FgDesc->GetUAV(*m_RDGRigidDynamics);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGRigidColliders)
            .AddReadWriteResource(*m_RDGRigidDynamics);
    }

    void MPMSimulatorPrivateData::PbMpmRigidIntegrate(FrameGraphBuilder& builder, f32 dt)
    {
        // shader from rigid simulator
        struct PushConst
        {
            Vector4f        m_Gravity;
            Vector4f        m_ExternalMoment;
            u32             m_NumRigidBodies;
            f32             m_DeltaTime;
            RHI::RhiSRVDesc m_IndirectRigidCounter; // Left 0
            RHI::RhiSRVDesc m_RigidColliders;
            RHI::RhiUAVDesc m_RigidDynamics;
        } pc{};

        pc.m_Gravity              = Vector4f(m_Config->m_Gravity, 0.0f);
        pc.m_ExternalMoment       = Vector4f(0.0f);
        pc.m_NumRigidBodies       = m_SceneData->m_NumGpuColliders;
        pc.m_DeltaTime            = dt;
        pc.m_IndirectRigidCounter = ~0u;

        int      numTGX = Math::DivRoundUp(m_SceneData->m_NumGpuColliders, IfritShader::Artemis::Rigid::kRigidTGSizeX);
        Vector3i workGroup = Vector3i(numTGX, 1, 1);

        Vec<String> extra;
        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
        {
            extra.push_back("IFSHADER_RIGID_DYNAMICS_3D");
        }

        AddComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidIntegrate",
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.RigidPostStateUpdateCS, extra), workGroup, pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_RigidColliders = ctx.m_FgDesc->GetSRV(*m_RDGRigidColliders);
                pc.m_RigidDynamics  = ctx.m_FgDesc->GetUAV(*m_RDGRigidDynamics);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGRigidColliders)
            .AddReadWriteResource(*m_RDGRigidDynamics);
    }

    void MPMSimulatorPrivateData::PbMpmRigidCollectCollisionPairs(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32             m_NumParticles;
            RHI::RhiSRVDesc m_ParticleCounter;
            RHI::RhiUAVDesc m_CollisionPairs;
            RHI::RhiUAVDesc m_CollisionPairCounter;
            RHI::RhiSRVDesc m_ParticlePositions;
            RHI::RhiSRVDesc m_ParticleDisplacements;
            RHI::RhiSRVDesc m_RigidColliders;
            RHI::RhiSRVDesc m_RigidDynamics;
            RHI::RhiUAVDesc m_ParticleColor;
        } pc{};

        pc.m_NumParticles = ~0u;
        int NumRigids     = m_SceneData->m_NumGpuColliders;
        // TODO: another indirect dispatch buffer required for (rigid count>1)

        AddIndirectComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidCollectCollisionPairs",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMRigidCollectCollisionPairsCS), *m_RDGParticleCount,
            sizeof(u32), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounter       = ctx.m_FgDesc->GetSRV(*m_RDGParticleCount);
                pc.m_CollisionPairs        = ctx.m_FgDesc->GetUAV(*m_RDGRigidContactList);
                pc.m_CollisionPairCounter  = ctx.m_FgDesc->GetUAV(*m_RDGRigidContactCounter);
                pc.m_ParticlePositions     = ctx.m_FgDesc->GetSRV(*m_RDGParticlePosition);
                pc.m_ParticleDisplacements = ctx.m_FgDesc->GetSRV(*m_RDGParticleVelocity);
                pc.m_RigidColliders        = ctx.m_FgDesc->GetSRV(*m_RDGRigidColliders);
                pc.m_RigidDynamics         = ctx.m_FgDesc->GetSRV(*m_RDGRigidDynamics);
                pc.m_ParticleColor         = ctx.m_FgDesc->GetUAV(*m_RDGParticleColor);
                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleCount)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleVelocity)
            .AddReadResource(*m_RDGRigidColliders)
            .AddReadResource(*m_RDGRigidDynamics)
            .AddReadWriteResource(*m_RDGRigidContactList)
            .AddReadWriteResource(*m_RDGRigidContactCounter)
            .AddReadWriteResource(*m_RDGParticleColor);
    }

    void MPMSimulatorPrivateData::PbMpmRigidResetContactCounter(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            RHI::RhiUAVDesc m_CollisionPairCounter;
            RHI::RhiUAVDesc m_BoundaryPairCounter;
        } pc{};

        AddComputePass<PushConst>(builder, "MPMSimulator.PbMpmRigidResetContactCounter",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMRigidResetCollisionPairCounterCS), Vector3i(1, 1, 1),
            pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_CollisionPairCounter = ctx.m_FgDesc->GetUAV(*m_RDGRigidContactCounter);
                pc.m_BoundaryPairCounter  = ctx.m_FgDesc->GetUAV(*m_RDGRigidBoundaryContactCounter);
                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGRigidContactCounter)
            .AddReadWriteResource(*m_RDGRigidBoundaryContactCounter);
    }

    void MPMSimulatorPrivateData::ParticleDrainAll(FrameGraphBuilder& builder)
    {
        struct PushConst
        {
            u32 m_ParticleCounter;

        } pc{};

        AddComputePass<PushConst>(builder, "MPMSimulator.ParticleDrainAll",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMParticleDrainAllCS), Vector3i(1, 1, 1), pc,
            [this](PushConst pc, const FrameGraphPassContext& ctx) {
                pc.m_ParticleCounter = ctx.m_FgDesc->GetUAV(*m_RDGParticleCount);
                SetRootConstant(pc, ctx);
            })
            .AddReadWriteResource(*m_RDGParticleCount);
    }

    void MPMSimulatorPrivateData::ParticleEmit(
        FrameGraphBuilder& builder, const MPMParticleEmitArgs& args, u32 numParticles)
    {
        struct PushConst
        {
            Vector4f m_EmitColor;
            f32      m_DefaultYoungs;
            f32      m_DefaultPossion;
            i32      m_DefaultMatType;

            i32      m_NumParticles;
            f32      m_Mass;
            f32      m_Density;
            u32      m_Grid;
            u32      m_ParticleLocationSrc;

            u32      m_ParticleLocation;
            u32      m_ParticleVelocity;
            u32      m_ParticleMass;
            u32      m_ParticleVolume;
            u32      m_ParticleB;
            u32      m_ParticleDeformationGrad;
            u32      m_ParticleDeformationGradDet;
            u32      m_ParticleMaterial;
            u32      m_ParticleLiquidDensity;
            u32      m_ParticleCounter;
            u32      m_ParticleColor;
        } pc;
        pc.m_EmitColor      = args.m_EmitColor;
        pc.m_DefaultYoungs  = args.m_YoungsModulus;
        pc.m_DefaultPossion = args.m_PoissonRatio;
        pc.m_DefaultMatType = static_cast<i32>(args.m_MaterialType);
        pc.m_NumParticles   = static_cast<i32>(numParticles);
        pc.m_Mass           = args.m_Mass;
        pc.m_Density        = args.m_Density;

        i32 tgX = DivRoundUp(numParticles, kDefaultTGX);

        AddComputePass<PushConst>(builder, "MPMSimulator.ParticleEmit",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMParticleEmitCS), Vector3i(tgX, 1, 1), pc,
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
                pc.m_ParticleColor              = ctx.m_FgDesc->GetUAV(*m_RDGParticleColor);
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
            .AddReadWriteResource(*m_RDGParticleColor)
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMPbMpmParticleIntegrateCS), *m_RDGParticleCount,
            sizeof(u32), pc,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMPbMpmResolveConstraintsCS), *m_RDGParticleCount,
            sizeof(u32), pc,
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
            u32 m_ParticleColor;
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMParticleInitCS), Vector3i(tgX, 1, 1), pc,
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
                pc.m_ParticleColor              = ctx.m_FgDesc->GetUAV(*m_RDGParticleColor);

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
            .AddWriteResource(*m_RDGParticleLiquidDensity)
            .AddWriteResource(*m_RDGParticleCount)
            .AddWriteResource(*m_RDGParticleColor)
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

        if (firstFrame)
        {
            extra.push_back("IFSHADER_MPM_GRIDRESET_INIT");
            AddComputePass<PushConst>(builder, "MPMSimulator.GridReset.Init",
                GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMGridResetCS, extra), Vector3i(tgX, 1, 1), pc,
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
                GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMGridResetCS, extra), *m_RDGValidGridCounter,
                sizeof(u32), pc,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMP2GCS, extra), *m_RDGParticleCount, sizeof(u32), pc,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMGridRegularizeCS), Vector3i(tgX, 1, 1), pc,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMGridForceUpdateCS), *m_RDGParticleCount, sizeof(u32),
            pc,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMGridGravityApplyCS), *m_RDGValidGridCounter,
            sizeof(u32), pc,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMGridVelocityUpdateCS, extra), *m_RDGValidGridCounter,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMParticleAdvectionCS), *m_RDGParticleCount,
            sizeof(u32), pc,
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
        auto stagedIndex    = RHI->CreateStagedSingleBuffer(m_ParticleData->m_ParticleIndex.get());
        auto stagedGridAttr = RHI->CreateStagedSingleBuffer(m_GridAttribute.get());
        auto stagedPosition = RHI->CreateStagedSingleBuffer(m_ParticleData->m_ParticlePosition.get());
        auto stagedCounter  = RHI->CreateStagedSingleBuffer(m_ParticleData->m_ParticleCount.get());
        tq->RunSyncCommand([&](const RhiCommandList* cmd) {
            stagedIndex->CmdCopyToDevice(
                cmd, particleIndexData.data(), SizeCast<u32>(particleIndexData.size() * sizeof(u32)), 0);
            stagedGridAttr->CmdCopyToDevice(cmd, &gridAttr, sizeof(MPMSimulatorGridAttribute), 0);
            stagedCounter->CmdCopyToDevice(
                cmd, particleDataSection.data(), SizeCast<u32>(particleDataSection.size() * sizeof(u32)), 0);

            if (m_HasInitParticleLocations)
            {
                if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
                {
                    if (std::holds_alternative<Vec<Vector2f>>(m_InitParticleLocations))
                    {
                        auto& initLocs = std::get<Vec<Vector2f>>(m_InitParticleLocations);
                        stagedPosition->CmdCopyToDevice(
                            cmd, initLocs.data(), SizeCast<u32>(initLocs.size() * sizeof(Vector2f)), 0);
                    }
                    else
                    {
                        IF_LOG_ASSERTION("Artemis.MPM", false,
                            "MPMSimulator: Initial particle locations must be of type Vec<Vector2f> "
                            "for 2D simulations.");
                    }
                }
                else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
                {
                    if (std::holds_alternative<Vec<Vector4f>>(m_InitParticleLocations))
                    {
                        auto& initLocs = std::get<Vec<Vector4f>>(m_InitParticleLocations);
                        stagedPosition->CmdCopyToDevice(
                            cmd, initLocs.data(), SizeCast<u32>(initLocs.size() * sizeof(Vector4f)), 0);
                    }
                    else
                    {
                        IF_LOG_ASSERTION("Artemis.MPM", false,
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
            GetShader(Runtime::Internal::kIntShaderTableArtemis.MPMG2PCS, extra), *m_RDGParticleCount, sizeof(u32), pc,
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
        IF_LOG_ASSERTION(
            "Artemis.MPM", m_Config, "MPMSimulator: Config must be set before getting the number of grids.");
        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
            return m_Config->m_GridSize.x * m_Config->m_GridSize.y;
        else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
            return m_Config->m_GridSize.x * m_Config->m_GridSize.y * m_Config->m_GridSize.z;
        return 0;
    }

    template <u32 Dimension> void MPMSimulatorPrivateData::HandleManualParticleEmit(FrameGraphBuilder& builder)
    {
        auto rhi    = builder.GetRhi();
        using EleTp = MPMSimulatorTypes<Dimension>::FSpatialVectorAligned;
        using VecTp = Vec<EleTp>;
        for (auto& emitRequest : m_EmissionInfos)
        {
            IF_LOG_ASSERTION("Artemis.MPM", m_EmissionInfos.size() == 1,
                "MPMSimulator: Manual particle emission is only supported for a single emission request at a time.");
            if (std::holds_alternative<VecTp>(emitRequest.m_InitialParticleLocations))
            {
                auto  stagedLocation = rhi->CreateStagedSingleBuffer(m_ParticleEmitLocations.get());
                auto  tq             = rhi->GetQueue(RhiQueueCapability::RhiQueue_Transfer);
                auto& locations      = std::get<VecTp>(emitRequest.m_InitialParticleLocations);
                tq->RunSyncCommand([&](const RhiCommandList* cmd) {
                    stagedLocation->CmdCopyToDevice(
                        cmd, locations.data(), SizeCast<u32>(locations.size() * sizeof(EleTp)), 0);
                });
                ParticleEmit(builder, emitRequest.m_EmissionArgs, SizeCast<u32>(locations.size()));
                m_Config->m_DefaultNumParticles += static_cast<u32>(locations.size());
            }
            else
            {
                IF_LOG_ASSERTION("Artemis.MPM", false, "MPMSimulator: particle dimension mismatches.");
            }
        }
        m_EmissionInfos.clear();
    }

    template <u32 Dimension> void MPMSimulatorPrivateData::InitGPUResources(RhiBackend* RHI)
    {
        using MTypes = MPMSimulatorTypes<Dimension>;
        IF_LOG_ASSERTION(
            "Artemis.MPM", m_Config, "MPMSimulator: Config must be set before initializing GPU resources.");
        static_assert(Dimension == 2 || Dimension == 3, "MPMSimulator: Invalid dimension specified.");

        auto numParticles = m_Config->m_MaxParticles;
        auto numGrids     = GetNumGrids();

        auto particleCountSz = SizeCast<u32>(sizeof(u32) * 4);

        auto particlePosSz           = numParticles * MTypes::kFSpatialVectorAlignedSize;
        auto particleColorSz         = SizeCast<u32>(numParticles * sizeof(Vector4f));
        auto particleVelSz           = numParticles * MTypes::kFSpatialVectorAlignedSize;
        auto particleMassSz          = numParticles * MTypes::kFScalarSize;
        auto particleDeformGradSz    = numParticles * MTypes::kFSpatialTransformAlignedSize;
        auto particleJSz             = numParticles * MTypes::kFScalarSize;
        auto particleVolSz           = numParticles * MTypes::kFScalarSize;
        auto particleApicBSz         = SizeCast<u32>(numParticles * MTypes::kFSpatialTransformAlignedSize);
        auto particleIndexSz         = SizeCast<u32>(numParticles * sizeof(u32));
        auto particleDebugSz         = SizeCast<u32>(numParticles * 64);
        auto particleStressContribSz = SizeCast<u32>(numParticles * MTypes::kFSpatialTransformAlignedSize);
        auto particleMatPropertySz   = SizeCast<u32>(numParticles * sizeof(MPMParticleMaterials));
        auto particleLiquidSz        = SizeCast<u32>(numParticles * sizeof(f32));

        auto gridForceSz = numGrids * MTypes::kFSpatialVectorAlignedSize;
        auto gridVelSz   = numGrids * MTypes::kFSpatialVectorAlignedSize;
        auto gridMassSz  = numGrids * MTypes::kFScalarSize;
        auto gridAttrSz  = SizeCast<u32>(sizeof(MPMSimulatorGridAttribute));

        auto contactIndSz = SizeCast<u32>(sizeof(u32) * 4);
        auto contactSz = SizeCast<u32>(sizeof(Shader::Artemis::FMPMRigidCouplingContactPair) * m_Config->m_MaxContacts);
        auto boundaryIndSz = SizeCast<u32>(sizeof(u32) * 4);
        auto boundarySz =
            SizeCast<u32>(sizeof(Shader::Artemis::FMPMRigidBoundaryContactPair) * m_Config->m_MaxContacts);

        auto inddrawSz = sizeof(u32) * 4;

        auto defaultUsage  = RhiBufferUsage::RhiBufferUsage_SSBO | RhiBufferUsage::RhiBufferUsage_CopyDst;
        auto indexUsage    = defaultUsage | RhiBufferUsage::RhiBufferUsage_Index;
        auto indirectUsage = defaultUsage | RhiBufferUsage::RhiBufferUsage_Indirect;

        m_ParticleData->m_ParticleCount =
            RHI->CreateBufferDevice("MPM_ParticleCount", particleCountSz, indirectUsage, true);
        m_ParticleData->m_ParticleColor =
            RHI->CreateBufferDevice("MPM_ParticleColor", particleColorSz, defaultUsage, true);
        m_ParticleData->m_ParticlePosition =
            RHI->CreateBufferDevice("MPM_ParticlePosition", particlePosSz, defaultUsage, true);
        m_ParticleEmitLocations =
            RHI->CreateBufferDevice("MPM_ParticleEmitLocations", particlePosSz, defaultUsage, true);
        m_ParticleData->m_ParticleVelocity =
            RHI->CreateBufferDevice("MPM_ParticleVelocity", particleVelSz, defaultUsage, true);
        m_ParticleData->m_ParticleMass =
            RHI->CreateBufferDevice("MPM_ParticleMass", particleMassSz, defaultUsage, true);
        m_ParticleData->m_ParticleDeformGrad =
            RHI->CreateBufferDevice("MPM_ParticleDeformGradient", particleDeformGradSz, defaultUsage, true);
        m_ParticleData->m_ParticleDeformGradDet =
            RHI->CreateBufferDevice("MPM_ParticleDeformGradientDeterminant", particleJSz, defaultUsage, true);
        m_ParticleData->m_ParticleVolume =
            RHI->CreateBufferDevice("MPM_ParticleVolume", particleVolSz, defaultUsage, true);
        m_ParticleData->m_ParticleApicB =
            RHI->CreateBufferDevice("MPM_ParticleApicB", particleApicBSz, defaultUsage, true);
        m_ParticleData->m_ParticleIndex =
            RHI->CreateBufferDevice("MPM_ParticleIndex", particleIndexSz, indexUsage, true);
        m_ParticleData->m_ParticleDebug =
            RHI->CreateBufferDevice("MPM_ParticleDebug", particleDebugSz, defaultUsage, true);
        m_ParticleData->m_ParticleStressContrib =
            RHI->CreateBufferDevice("MPM_ParticleStressContrib", particleStressContribSz, defaultUsage, true);
        m_ParticleData->m_ParticleMatProperty =
            RHI->CreateBufferDevice("MPM_ParticleMaterialProperty", particleMatPropertySz, defaultUsage, true);
        m_ParticleData->m_ParticleLiquidDensity =
            RHI->CreateBufferDevice("MPM_ParticleLiquiddDensity", particleLiquidSz, defaultUsage, true);

        m_GridForce     = RHI->CreateBufferDevice("MPM_GridForce", gridForceSz, defaultUsage, true);
        m_GridVelocity  = RHI->CreateBufferDevice("MPM_GridVelocity", gridVelSz, defaultUsage, true);
        m_GridMass      = RHI->CreateBufferDevice("MPM_GridMass", gridMassSz, defaultUsage, true);
        m_GridAttribute = RHI->CreateBufferDevice("MPM_GridAttribute", gridAttrSz, defaultUsage, true);

        m_RigidContactCounter = RHI->CreateBufferDevice("MPM_RigidContactCounter", contactIndSz, indirectUsage, true);
        m_RigidContactList    = RHI->CreateBufferDevice("MPM_RigidContactList", contactSz, defaultUsage, true);
        m_RigidBoundaryContactCounter =
            RHI->CreateBufferDevice("MPM_BoundaryContactCounter", boundaryIndSz, indirectUsage, true);
        m_RigidBoundaryContactList = RHI->CreateBufferDevice("MPM_BoundaryContactList", boundarySz, defaultUsage, true);

        m_RenderParticleIndDrawBuffer =
            RHI->CreateBufferDevice("MPM_RenderParticleIndDraw", SizeCast<u32>(inddrawSz), indirectUsage, true);

        PrepareInitialGPUData(RHI);
    }

    void MPMSimulatorPrivateData::InitRDGResources(FrameGraphBuilder& builder)
    {
        // Persistent
        m_RDGParticleCount    = &builder.ImportBuffer("MPM_ParticleCount", m_ParticleData->m_ParticleCount.get());
        m_RDGParticleColor    = &builder.ImportBuffer("MPM_ParticleColor", m_ParticleData->m_ParticleColor.get());
        m_RDGParticlePosition = &builder.ImportBuffer("MPM_ParticlePosition", m_ParticleData->m_ParticlePosition.get());
        m_RDGParticleEmitLocations = &builder.ImportBuffer("MPM_ParticleEmitLocations", m_ParticleEmitLocations.get());
        m_RDGParticleVelocity = &builder.ImportBuffer("MPM_ParticleVelocity", m_ParticleData->m_ParticleVelocity.get());
        m_RDGParticleMass     = &builder.ImportBuffer("MPM_ParticleMass", m_ParticleData->m_ParticleMass.get());
        m_RDGParticleDeformGrad =
            &builder.ImportBuffer("MPM_ParticleDeformGradient", m_ParticleData->m_ParticleDeformGrad.get());
        m_RDGParticleDeformGradDet = &builder.ImportBuffer(
            "MPM_ParticleDeformGradientDeterminant", m_ParticleData->m_ParticleDeformGradDet.get());
        m_RDGParticleVolume = &builder.ImportBuffer("MPM_ParticleVolume", m_ParticleData->m_ParticleVolume.get());
        m_RDGParticleApicB  = &builder.ImportBuffer("MPM_ParticleApicB", m_ParticleData->m_ParticleApicB.get());
        m_RDGParticleDebug  = &builder.ImportBuffer("MPM_ParticleDebug", m_ParticleData->m_ParticleDebug.get());
        m_RDGParticleStressContrib =
            &builder.ImportBuffer("MPM_ParticleStressContrib", m_ParticleData->m_ParticleStressContrib.get());
        m_RDGParticleMatProperty =
            &builder.ImportBuffer("MPM_ParticleMaterialProperty", m_ParticleData->m_ParticleMatProperty.get());
        m_RDGParticleLiquidDensity =
            &builder.ImportBuffer("MPM_ParticleLiquidDensity", m_ParticleData->m_ParticleLiquidDensity.get());

        m_RDGGridForce     = &builder.ImportBuffer("MPM_GridForce", m_GridForce.get());
        m_RDGGridVelocity  = &builder.ImportBuffer("MPM_GridVelocity", m_GridVelocity.get());
        m_RDGGridMass      = &builder.ImportBuffer("MPM_GridMass", m_GridMass.get());
        m_RDGGridAttribute = &builder.ImportBuffer("MPM_GridAttribute", m_GridAttribute.get());

        m_RDGRenderParticleIndDrawBuffer =
            &builder.ImportBuffer("MPM_RenderParticleIndDraw", m_RenderParticleIndDrawBuffer.get());

        m_RDGRigidContactCounter = &builder.ImportBuffer("MPM_RigidContactCounter", m_RigidContactCounter.get());
        m_RDGRigidContactList    = &builder.ImportBuffer("MPM_RigidContactList", m_RigidContactList.get());
        m_RDGRigidBoundaryContactCounter =
            &builder.ImportBuffer("MPM_BoundaryContactCounter", m_RigidBoundaryContactCounter.get());
        m_RDGRigidBoundaryContactList =
            &builder.ImportBuffer("MPM_BoundaryContactList", m_RigidBoundaryContactList.get());

        if (m_DebugRenderTarget)
        {
            m_RDGRenderTarget = &builder.ImportTexture("MPM_DebugRenderTarget", m_DebugRenderTarget);
        }

        // PBMPM Rigid Coupling
        if (m_ShouldIntegrateRigids)
        {
            m_RDGRigidColliders =
                &builder.ImportBuffer("MPM_RigidColliders", m_SceneData->m_GpuColliderDataBuffer.get());
            m_RDGRigidDynamics =
                &builder.ImportBuffer("MPM_RigidDynamics", m_SceneData->m_GpuColliderDataBufferRuntime.get());
        }

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
                IF_LOG_ASSERTION("Artemis.MPM", false, "MPMSimulator: Invalid problem dimension specified.");
        }
        InitRDGResources(builder);
        if (isFirstRun) IF_UNLIKELY
        {
            ParticleInit(builder);
        }

        if (m_Config->m_Dimension == MPMSimulatorProblemDimension::TwoDimensional)
            HandleManualParticleEmit<2>(builder);
        else if (m_Config->m_Dimension == MPMSimulatorProblemDimension::ThreeDimensional)
            HandleManualParticleEmit<3>(builder);

        if (m_HasGlobalDrain)
        {
            ParticleDrainAll(builder);
            m_HasGlobalDrain = false;
        }

        f32 deltaTimePerSubstep = deltaTime / static_cast<f32>(m_Config->m_Substeps);
        if (isPbMpm)
        {
            if (m_ShouldIntegrateRigids)
            {
                PbMpmRigidLoadTransform(builder);
            }
            for (auto i = 0u; i < m_Config->m_Substeps; ++i)
            {
                {
                    IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.PbMpmSubstep");
                    if (m_ShouldIntegrateRigids)
                    {
                        PbMpmRigidResetContactCounter(builder);
                        PbMpmRigidCollectCollisionPairs(builder);
                        PbMpmRigidCollectBoundaryContactPairs(builder);
                    }
                    PbMpmRigidLoadTransform(builder);
                    for (auto j = 0u; j < m_Config->m_PbMpmIterations; ++j)
                    {
                        bool isLastIteration  = (j == m_Config->m_PbMpmIterations - 1);
                        bool isFirstIteration = (j == 0);
                        u32  firstOrLastRun   = 0;
                        firstOrLastRun |= (isFirstIteration) ? 1 : 0;
                        firstOrLastRun |= (isLastIteration) ? 2 : 0;
                        {
                            IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.PbMpmIteration");

                            GridReset(builder, isFirstRun, isFirstIteration);
                            PbMpmResolveConstraints(builder, deltaTimePerSubstep);
                            if (m_ShouldIntegrateRigids)
                            {
                                PbMpmRigidContactConstraintResolve(builder);
                                PbMpmRigidBoundaryConstraintResolve(builder);
                            }

                            ParticleToGridTransfer(builder, deltaTimePerSubstep, firstOrLastRun);
                            if (isFirstIteration)
                            {
                                GridVelocityNormalize(builder);
                            }
                            GridVelocityUpdate(builder, deltaTimePerSubstep, isFirstIteration);
                            GridToParticleTransfer(builder, deltaTimePerSubstep);
                        }
                        isFirstRun = false;
                    }
                    PbMpmParticleIntegrate(builder, deltaTimePerSubstep);
                    if (m_ShouldIntegrateRigids)
                    {
                        PbMpmRigidIntegrate(builder, deltaTimePerSubstep);
                        PbMpmRigidSyncTransform(builder);
                    }
                }
            }
        }
        else
        {
            for (auto i = 0u; i < m_Config->m_Substeps; ++i)
            {
                {
                    IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.MpmSubstep");
                    GridReset(builder, isFirstRun, true);
                    ParticleToGridTransfer(builder, deltaTimePerSubstep, false);
                    GridVelocityNormalize(builder);
                    GridForceUpdate(builder);
                    GridGravityApply(builder);
                    GridVelocityUpdate(builder, deltaTimePerSubstep, false);
                    GridToParticleTransfer(builder, deltaTimePerSubstep);
                    ParticleAdvect(builder, deltaTimePerSubstep);
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
        IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.ParticleRender");

        struct PushConst_PrepareInd
        {
            u32 m_CounterId;
            u32 m_IndirectDrawId;
        } pci{};
        AddComputePass<PushConst_PrepareInd>(builder, "MPMSimulator.ParticleRenderPrepareIndirect",
            GetShader(Runtime::Internal::kIntShaderTableArtemis.ParticleIndDrawBufferPrepCS), Vector3i(1, 1, 1), pci,
            [this](PushConst_PrepareInd pc, const FrameGraphPassContext& ctx) {
                pc.m_CounterId      = ctx.m_FgDesc->GetSRV(*m_RDGParticleCount);
                pc.m_IndirectDrawId = ctx.m_FgDesc->GetUAV(*m_RDGRenderParticleIndDrawBuffer);

                SetRootConstant(pc, ctx);
            })
            .AddReadResource(*m_RDGParticleCount)
            .AddWriteResource(*m_RDGRenderParticleIndDrawBuffer);

        struct PushConst
        {
            u32 m_PositionId;
            u32 m_ColorId;
            f32 m_GridRange;
            f32 m_AspectRatio;
            f32 m_PointSize;
        };

        auto& pass = builder.AddGraphicsPass("MPMSimulator.ParticleRender",
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.ParticleRender2dVS, {}),
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.ParticleRender2dFS, {}),
            GetPushConstSize<PushConst>(), RhiRasterizerTopology::Point);

        pass.SetExecutionFunction([renderTarget, this](const FrameGraphPassContext& ctx) {
            auto      rt = renderTarget;

            auto      cmd      = ctx.m_CmdList;
            auto      rtWidth  = rt->GetWidth();
            auto      rtHeight = rt->GetHeight();

            PushConst pc;
            pc.m_PositionId  = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
            pc.m_ColorId     = ctx.m_FgDesc->GetUAV(*m_RDGParticleColor);
            pc.m_GridRange   = m_Config->m_GridSize.x * m_Config->m_GridSpacing;
            pc.m_AspectRatio = (f32)rtWidth / (f32)rtHeight;
            pc.m_PointSize   = m_ParticleRenderSize;

            cmd->AttachIndexBuffer(m_ParticleData->m_ParticleIndex.get());
            cmd->SetCullMode(RhiCullMode::None);
            cmd->SetPushConst(&pc, 0, sizeof(PushConst));
            // cmd->DrawIndexed(m_Config->m_DefaultNumParticles, 1, 0, 0, 0);
            cmd->DrawIndirect(m_RenderParticleIndDrawBuffer.get(), 0);
        });
        pass.AddRenderTarget(*renderTarget, RHI::RhiRenderTargetLoadOp::Load)
            .AddReadResource(*m_RDGRenderParticleIndDrawBuffer)
            .AddReadResource(*m_RDGParticlePosition)
            .AddReadResource(*m_RDGParticleColor);
    }

    void MPMSimulatorPrivateData::ParticleRender3D(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.ParticleRender");

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
        };

        auto& pass = builder.AddGraphicsPass("MPMSimulator.ParticleRender3D",
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.ParticleRender3dVS, {}),
            ShaderVariantDesc(Runtime::Internal::kIntShaderTableArtemis.ParticleRender3dFS, {}),
            GetPushConstSize<PushConst>(), RhiRasterizerTopology::Point);

        pass.SetExecutionFunction([renderTarget, this, mvp](const FrameGraphPassContext& ctx) {
            auto      rt = renderTarget;

            auto      cmd      = ctx.m_CmdList;
            auto      rtWidth  = rt->GetWidth();
            auto      rtHeight = rt->GetHeight();

            PushConst pc;
            pc.m_PositionId = ctx.m_FgDesc->GetUAV(*m_RDGParticlePosition);
            pc.m_MVP        = mvp;

            cmd->AttachIndexBuffer(m_ParticleData->m_ParticleIndex.get());
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

    IFRIT_APIDECL MPMSimulatorConfig& MPMSimulator::GetActiveConfig() { return m_Config; }

    IFRIT_APIDECL void                MPMSimulator::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        if (!m_Data->m_ParticleData)
        {
            return;
        }

        {
            IFRIT_FRAMEGRAPH_EVENT_SCOPE(builder, "MPMSimulator.SolverStep");
            m_Data->RunSolverStep(builder, deltaTime);
        }

        if (m_Data->m_DebugRenderTarget)
        {
            Render(builder, m_Data->m_RDGRenderTarget);
        }
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
            IF_LOG_ASSERTION("Artemis.MPM", false, "MPMSimulator: Invalid problem dimension specified for rendering.");
        }
    }

    template <u32 Dimension IF_REQUIRES(Dimension == 2 || Dimension == 3)>
    void MPMSimulator::SetInitParticleLocations(const Vec<TGenericVector<f32, Dimension>>& locations)
    {
        IF_CONSTEXPR auto TAlignedDim = Dimension + (Dimension == 3 ? 1 : 0);
        using TAlignedVec             = TGenericVector<f32, TAlignedDim>;

        IF_LOG_ASSERTION("Artemis.MPM", locations.size() > 0, "MPMSimulator: No particles found");

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

    template IFRIT_APIDECL void MPMSimulator::EmitParticles<2>(
        const Vec<TGenericVector<f32, 2>>& locations, const MPMParticleEmitArgs& args);
    template IFRIT_APIDECL void MPMSimulator::EmitParticles<3>(
        const Vec<TGenericVector<f32, 3>>& locations, const MPMParticleEmitArgs& args);

    IFRIT_APIDECL RHI::RhiBufferRef MPMSimulator::GetParticlePositionBuffer()
    {
        return m_Data->m_ParticleData->m_ParticlePosition;
    }
    IFRIT_APIDECL RHI::RhiBufferRef MPMSimulator::GetParticleCounterBuffer()
    {
        return m_Data->m_ParticleData->m_ParticleCount;
    }

    IFRIT_APIDECL void MPMSimulator::RequestClearParticles() { m_Data->m_HasGlobalDrain = true; }
    void               MPMSimulator::SetDefaultSize(f32 size) { m_Data->m_ParticleRenderSize = size; }

    IFRIT_APIDECL void MPMSimulator::CollectScene(Scene* scene)
    {
        m_Data->m_FrameId++;

        // Emitters
        auto emitters = scene->FilterObjects([](GameObject* obj) {
            auto emitter = obj->GetComponent<MPMParticleEmitter>();
            return emitter != nullptr && emitter->IsEnabled();
        });
        for (auto* emitter : emitters)
        {
            auto& emitterComponent = *emitter->GetComponent<MPMParticleEmitter>();
            if (emitterComponent.ShouldEmitParticle(m_Data->m_FrameId))
            {
                auto particleLoc = emitterComponent.GetEmitParticlePosition2D();
                auto emitArgs    = emitterComponent.GetEmitArgs();
                EmitParticles<2>(particleLoc, emitArgs);
            }
        }

        // Particle Containers
        auto containers = scene->FilterObjects([](GameObject* obj) {
            auto container = obj->GetComponent<MPMParticleContainer>();
            return container != nullptr && container->IsEnabled();
        });
        IF_LOG_ASSERTION("Artemis.MPM", containers.size() <= 1,
            "MPMSimulator: Multiple MPMParticleContainer components found in the scene. "
            "Only one is allowed at a time.");

        if (containers.size() != 0)
        {
            auto& containerComponent = *containers[0]->GetComponent<MPMParticleContainer>();
            m_Data->m_ParticleData   = containerComponent.GetDeviceData();
            if (!containerComponent.GetIsDeviceDataReady())
            {
                m_Data->m_RebuildGPUResources = true;
                containerComponent.SetIsDeviceDataReady(true);
            }
        }
        else IF_UNLIKELY
        {
            m_Data->m_ParticleData = nullptr;
            IF_LOG_WARNING("Artemis.MPM",
                "MPMSimulator: No MPMParticleContainer found in the scene. "
                "Please add one to manage particle data.");
        }

        // Rigid Coupling
        if (m_Data->m_Config->m_EnableRigidCoupling)
        {
            m_Data->m_ShouldIntegrateRigids = true;

            auto perFrameData = scene->GetPerFrameData();
            if (perFrameData->m_ExtraData.count(Internal::kArtemisSceneDataKey) == 0) IF_UNLIKELY
            {
                IF_LOG_ERROR("Artemis.MPM",
                    "MPMSimulator: No Artemis scene data found.Please ensure the scene is properly initialized.");
                m_Data->m_ShouldIntegrateRigids = false;
            }
            else
            {
                auto ptr            = perFrameData->m_ExtraData[Internal::kArtemisSceneDataKey].get();
                m_Data->m_SceneData = static_cast<ArtemisSceneData*>(ptr);
            }
        }
    }

    IFRIT_APIDECL void MPMSimulator::SetDebugRenderTarget(RHI::RhiTexture* rt) { m_Data->m_DebugRenderTarget = rt; }

} // namespace Ifrit::Runtime::Artemis