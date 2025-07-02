#include "ifrit/runtime/physics/siro/mpm/MPMSimulator.h"

namespace Ifrit::Runtime::Siro
{
    template <u32 Dimension> struct MPMSimulatorTypes;

    template <> struct MPMSimulatorTypes<2>
    {
        using FSpatialVector        = Vector2f;
        using FSpatialTransform     = Matrix2x2f;
        using FSpatialVectorAligned = Vector2f;
        using FScalar               = f32;
    };

    template <> struct MPMSimulatorTypes<3>
    {
        using FSpatialVector        = Vector3f;
        using FSpatialTransform     = Matrix3x3f;
        using FSpatialVectorAligned = Vector4f;
        using FScalar               = f32;
    };

    struct MPMSimulatorPrivateData
    {
    };

    IFRIT_APIDECL MPMSimulator::MPMSimulator() : m_Data(new MPMSimulatorPrivateData()) {}
    IFRIT_APIDECL MPMSimulator::~MPMSimulator()
    {
        delete m_Data;
        m_Data = nullptr;
    }
    IFRIT_APIDECL void MPMSimulator::SetConfig(const MPMSimulatorConfig& cfg) { m_Config = cfg; }

    IFRIT_APIDECL void MPMSimulator::RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime)
    {
        // TODO
    }

    IFRIT_APIDECL void MPMSimulator::Render(FrameGraphBuilder& builder, FGTextureNode* renderTarget)
    {
        // TODO
    }

} // namespace Ifrit::Runtime::Siro