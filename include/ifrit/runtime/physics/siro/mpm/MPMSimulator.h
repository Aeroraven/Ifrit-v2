#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/siro/SiroIntegrator.h"

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
        MLS
    };

    enum class MPMSimulatorParticleType : u8
    {
        Jelly = 0,
        Fluid = 1
    };

    struct MPMSimulatorConfig
    {
        MPMSimulatorTopologySource   m_TopoSource = MPMSimulatorTopologySource::Preset;
        MPMSimulatorProblemDimension m_Dimension  = MPMSimulatorProblemDimension::TwoDimensional;
        MPMSimulatorVariant          m_Variant    = MPMSimulatorVariant::MLS;

        Vector3u                     m_GridSize          = Vector3u(128, 128, 128);
        Vector3f                     m_GridOffset        = Vector3f(0.0f);
        Vector3u                     m_GridBoundaryWidth = Vector3u(3, 3, 3);
        Vector3f                     m_Gravity           = Vector3f(0.0f, -1.0f, 0.0f);
        f32                          m_GridSpacing       = 1.0f / 128;
        f32                          m_DefaultMass       = (0.5f / 128);
        f32                          m_DefaultDensity    = 1.0f;

        f32                          m_DefaultYoungsModulus = 10.0f;
        f32                          m_DefaultPoissonRatio  = 0.2f;
        u32                          m_DefaultNumParticles  = 9000;
        u32                          m_Substeps             = 10;
        MPMSimulatorParticleType     m_DefaultParticleType  = MPMSimulatorParticleType::Fluid;
    };

    class IFRIT_RUNTIME_API MPMSimulator : public ISiroSolver
    {
    public:
        MPMSimulator();
        ~MPMSimulator();

        void         SetConfig(const MPMSimulatorConfig& cfg);
        virtual void RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime) override;
        void         Render(FrameGraphBuilder& builder, FGTextureNode* renderTarget);

    private:
        MPMSimulatorPrivateData* m_Data = nullptr;
        MPMSimulatorConfig       m_Config;
    };
} // namespace Ifrit::Runtime::Siro