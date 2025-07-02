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

    struct MPMSimulatorConfig
    {
        MPMSimulatorTopologySource   m_TopoSource = MPMSimulatorTopologySource::Preset;
        MPMSimulatorProblemDimension m_Dimension  = MPMSimulatorProblemDimension::ThreeDimensional;
        MPMSimulatorVariant          m_Variant    = MPMSimulatorVariant::MLS;

        Vector3u                     m_GridSize          = Vector3u(64, 64, 64);
        Vector3f                     m_GridOffset        = Vector3f(0.0f);
        Vector3u                     m_GridBoundaryWidth = Vector3u(3, 3, 3);
        Vector3f                     m_Gravity           = Vector3f(0.0f, -1.0f, 0.0f);
        f32                          m_GridSpacing       = 0.2f;
        f32                          m_DefaultMass       = 1.0f;
        f32                          m_DefaultDensity    = 1.0f;

        f32                          m_DefaultNeoHookeanMu     = 50 / (2 * (1 + 0.3f));
        f32                          m_DefaultNeoHookeanLambda = 50 * 0.3f / ((1 + 0.3f) * (1 - (2 * 0.3f)));
        u32                          m_DefaultNumParticles     = 100000;
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