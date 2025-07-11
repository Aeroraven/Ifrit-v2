#pragma once

#include "ifrit/runtime/geometry/ProceduralMesh.h"

namespace Ifrit::Runtime::Geometry
{
    struct ParticleSurfaceProceduralMeshPrivateData;
    class IFRIT_RUNTIME_API ParticleSurfaceProceduralMesh : public ProceduralMesh
    {
    public:
        ParticleSurfaceProceduralMesh();
        ~ParticleSurfaceProceduralMesh();

        void UpdateMesh(FrameGraphBuilder& builder) override;
        void SetParticleData(RHI::RhiBufferRef particleDataBuffer, RHI::RhiBufferRef particleCount);

        void Init(RHI::RhiBackend* rhi, u32 maxParticles, u32 maxIndices, Vector4i gridSize, Vector3f minBound,
            Vector3f maxBound);

    private:
        ParticleSurfaceProceduralMeshPrivateData* m_Data = nullptr;
    };
} // namespace Ifrit::Runtime::Geometry