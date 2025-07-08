#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/siro/SiroIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"

namespace Ifrit::Runtime::Siro
{
    struct MPMSimulatorPrivateData;

    enum class MPMSimulatorTopologySource : u8
    {
        External,
        Preset
    };

    enum class MPMSimulatorProblemDimension : u8
    {
        TwoDimensional,
        ThreeDimensional,
    };

    enum class MPMSimulatorVariant : u8
    {
        NonMLS,
        MLS,
        PBMPM,
    };

    enum class MPMSimulatorParticleType : u8
    {
        Jelly = 0,
        Fluid = 1,
        Snow  = 2,
        Visco = 3
    };

    struct MPMSimulatorConfig
    {
        IF_CONSTEXPR static u32      kDefaultGridSizeX = 64;

        MPMSimulatorTopologySource   m_TopoSource = MPMSimulatorTopologySource::Preset;
        MPMSimulatorProblemDimension m_Dimension  = MPMSimulatorProblemDimension::TwoDimensional;
        MPMSimulatorVariant          m_Variant    = MPMSimulatorVariant::PBMPM;

        Vector3u                     m_GridSize   = Vector3u(kDefaultGridSizeX, kDefaultGridSizeX, kDefaultGridSizeX);
        Vector3f                     m_GridOffset = Vector3f(0.0f);
        Vector3u                     m_GridBoundaryWidth = Vector3u(3, 3, 3);
        Vector3f                     m_Gravity           = Vector3f(0.0f, -1.0f, 0.0f);
        f32                          m_GridSpacing       = 1.0f / kDefaultGridSizeX;
        f32                          m_DefaultMass       = (0.5f / kDefaultGridSizeX);
        f32                          m_DefaultDensity    = 1.0f;

        f32                          m_DefaultYoungsModulus   = 50.0f;
        f32                          m_DefaultPoissonRatio    = 0.2f;
        f32                          m_DefaultViscoPlasticity = 0.9f;
        u32                          m_DefaultNumParticles    = 2000;
        u32                          m_Substeps               = 5;
        MPMSimulatorParticleType     m_DefaultParticleType    = MPMSimulatorParticleType::Jelly;

        // PBMPM
        u32                          m_PbMpmIterations                           = 14;
        f32                          m_PbMpmDefaultElasticityInterpolationFactor = 0.001f;
        f32                          m_PbMpmDefaultElasticityRelaxationFactor    = 1.5f;
        f32                          m_PbMpmDefaultLiquidViscosity               = 0.000f;
        f32                          m_PbMpmDefaultLiquidRelaxation              = 1.5f;
    };

    class IFRIT_RUNTIME_API MPMSimulator : public ISiroSolver
    {
    public:
        MPMSimulator();
        ~MPMSimulator();

        void         SetConfig(const MPMSimulatorConfig& cfg);
        virtual void RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime) override;
        void         Render(FrameGraphBuilder& builder, FGTextureNode* renderTarget);

        template <u32 Dimension IF_REQUIRES(Dimension == 2 || Dimension == 3)>
        void SetInitParticleLocations(const Vec<TGenericVector<f32, Dimension>>& locations);

    private:
        MPMSimulatorPrivateData* m_Data = nullptr;
        MPMSimulatorConfig       m_Config;
    };
} // namespace Ifrit::Runtime::Siro