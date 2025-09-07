#pragma ifrit.multi_compile IFSHADER_SURFRECON_2D
#include "ifrit.shader.neo/Meshing/SurfaceRecon/SurfRecon.Common.hlsli"

namespace IfritShader{
namespace Meshing{
namespace SurfRecon{

    struct PushConst_T
    {   
        TStructuredBufferHandle<FSurfReconGrid> m_Grid; //SRV
        TAtomicRWStructuredBufferHandle<int> m_ActiveVerticesCounter;
        TRWStructuredBufferHandle<int> m_ActiveVerticesList;
        TRWStructuredBufferHandle<int> m_CellParticleIndices;
        TRWStructuredBufferHandle<float> m_VertexDensity;
        FSpatialVectorHandle m_ParticleLocation; //SRV
        float m_KernelRadius; //H
        float m_KernelScaler;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;

    float Cubic(float X)
    {
        return X * X * X;
    }

    float SPHSmoothKernel(float D,float H)
    {
        if(D>=0.0f && D<=H)
        {
            float KernNorm = 315.0f / (64.0f * kPI * pow(H, 9.0f));
            return max(0.0f, KernNorm * pow(H * H - D * D, 3.0f));
        }
        return 0.0f;
    }

    float SPHSmoothKernel2D(float D, float H)
    {
        if(D >= 0.0f && D <= H)
        {
            float KernNorm = 15.0f / (kPI * pow(H, 3.0f));
            return max(0.0f, KernNorm * (H - D) * (H - D));
        }
        return 0.0f;
    }

    float IsotropicKernel3D(float3 R, float H)
    {
        float D = length(R);
        H = H * PushConst.m_KernelScaler; 
        D = D * PushConst.m_KernelScaler;
        return SPHSmoothKernel(D/H, H)/Cubic(H);
    }
    float IsotropicKernel2D(float2 R, float H)
    {
        float D = length(R);
        H = H * PushConst.m_KernelScaler; 
        D = D * PushConst.m_KernelScaler;
        return SPHSmoothKernel2D(D/H, H)/Cubic(H);
    }
    

    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kSurfReconVertexDensityTGSz, 1, 1)
    void SurfReconVertexDensityCS(uint3 ThreadId:SV_DispatchThreadID,
                                uint3 GroupId:SV_GroupID,
                                uint3 GroupThreadId:SV_GroupThreadID)
    {
        FSurfReconGrid Grid = PushConst.m_Grid.Load(0);
        FSpatialIndex CellsMaxBound = Grid.GetGridNumCellsInAxis()-1;
        int TotalActiveVertices = PushConst.m_ActiveVerticesCounter.Load(0);
        if(ThreadId.x >= TotalActiveVertices) return;
        int VertexIdx = PushConst.m_ActiveVerticesList.Load(ThreadId.x);

        FSpatialIndex VertexCoord = Grid.GetCellVertexFromFlattenId(VertexIdx);
        FSpatialVector VertexLocation = Grid.GetCellVertexPosition(VertexCoord);
        FSpatialIndex MaxBound = min(CellsMaxBound,VertexCoord+ FSpatialIndex(ceil(PushConst.m_KernelRadius))-1);
        FSpatialIndex MinBound = max(FSpatialIndex(0),VertexCoord-FSpatialIndex(ceil(PushConst.m_KernelRadius)+1));
        
        float TotalDensity = 0.0f;
        for(int dx = MinBound.x; dx <= MaxBound.x; ++dx)
        {
            for(int dy = MinBound.y; dy <= MaxBound.y; ++dy)
            {
                int dz = 0;
#ifndef IFSHADER_SURFRECON_2D
                for(dz = MinBound.z; dz <= MaxBound.z; ++dz)
#endif
                {
                    FSpatialIndex GridCoord = ToSpatialIndex(dx, dy, dz);
                    int GridFlattenId = Grid.GetCellFlattenId(GridCoord);

                    int NumParticlesInCell = Grid.GetNumParticlesInCell(GridFlattenId);
                    for(int i = 0; i < NumParticlesInCell; ++i)
                    {
                        int ParticleIdx = PushConst.m_CellParticleIndices.Load(GridFlattenId * kSurfReconMaxParticlesPerCell + i);
                        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ParticleIdx);
                        FSpatialVector R = VertexLocation - ParticleLocation;

                        float Density = 0.0f;
                        #ifdef IFSHADER_SURFRECON_2D
                            Density = IsotropicKernel2D(R.xy, PushConst.m_KernelRadius);
                        #else
                            Density = IsotropicKernel3D(R.xyz, PushConst.m_KernelRadius);
                        #endif
                        TotalDensity += Density;
                    }                    
                }
            }
        }
        PushConst.m_VertexDensity.Store(TotalDensity, VertexIdx);

    }

}}}