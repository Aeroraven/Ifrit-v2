#pragma once

#include "ifrit/runtime/geometry/surfaceop/ParticleSurfaceProceduralMesh.h"

namespace Ifrit::Runtime::Artemis
{
    class IFRIT_RUNTIME_API ProceduralMPMMesh : public Geometry::ParticleSurfaceProceduralMesh
    {
        typedef Geometry::ParticleSurfaceProceduralMesh Super;

    public:
        ProceduralMPMMesh()          = default;
        virtual ~ProceduralMPMMesh() = default;
        void UpdateMesh(FrameGraphBuilder& builder) override;
    };
} // namespace Ifrit::Runtime::Artemis