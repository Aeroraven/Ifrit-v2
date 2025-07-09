#pragma ifrit.multi_compile IFSHADER_MPM_3D
#pragma ifrit.multi_compile IFSHADER_MPM_MLS

#include "ifrit.shader.neo/Siro/MPM/MPM.Common.hlsli"
#include "ifrit.shader.neo/Random/Random.Simplex2d.hlsli"
namespace IfritShader{
namespace Siro{
namespace MPM{

    struct PushConst_T
    {
        float4 m_Gravity;
        FParticleCounterHandle m_ParticleCounter;
        float m_DeltaTime;

        FDenseGridStructureHandle m_Grid;
        FSpatialVectorHandle m_ParticleLocation;
        FSpatialVectorHandle m_ParticleVelocity; // !!! Particle Displacement Indeed !!!
        FPackedMatrixHandle m_ParticleDeformationGrad;
        FPackedMatrixHandle m_ParticleB; 
        TRWStructuredBufferHandle<FMpmParticleMaterial> m_ParticleMaterial;
        FScalarHandle m_ParticleLiquidDensity;
        FPackedMatrixHandle m_ParticleDebug;

        float m_ViscoPlasticity;
    };

    IFSHADER_PUSHCONST PushConst_T PushConst;



    IFSHADER_ENTRY(IFSHADER_CS_ENTRY)
    IFSHADER_BLOCKSIZE(kMpmTGSizeX, 1, 1)
    void PbMpmParticleIntegrateCS(uint3 ThreadId:SV_DispatchThreadID,
                           uint3 GroupId:SV_GroupID,
                           uint3 GroupThreadId:SV_GroupThreadID)
    {
        if(ThreadId.x >= PushConst.m_ParticleCounter.GetNumParticles())
            return;

        FDenseGridStructure Grid = PushConst.m_Grid.Load();
        FScalar GridSpacing = Grid.GetGridSpacing();
        FSpatialVector GridMinBound = Grid.GetMinBound();
        FSpatialVector GridMaxBound = Grid.GetMaxBound();

        FScalar DInv = 0.0f;
#ifdef IFSHADER_MPM_INTERP_CUBIC
        DInv = 3.0f * rcp(GridSpacing * GridSpacing);
#else
        DInv = 4.0f * rcp(GridSpacing * GridSpacing);
#endif  
        FMpmParticleMaterial ParticleMaterial = PushConst.m_ParticleMaterial.Load(ThreadId.x);

        // Advect the particle FIRST!
        FSpatialVector ParticleLocation = PushConst.m_ParticleLocation.Load(ThreadId.x);
        FSpatialVector ParticleDisplacement = PushConst.m_ParticleVelocity.Load(ThreadId.x);
        FSpatialVector NewParticleLocation = ParticleLocation + ParticleDisplacement;
        NewParticleLocation = clamp(NewParticleLocation, GridMinBound, GridMaxBound);

        bool IsElastic = IsElasticMaterial(ParticleMaterial);
        bool IsLiquid = IsLiquidMaterial(ParticleMaterial);
        bool IsVisco = IsViscoMaterial(ParticleMaterial);

        FSpatialTransform ParticleB = PushConst.m_ParticleB.GetMatrix(ThreadId.x);
        FSpatialTransform ParticleC = ParticleB * DInv; // Deform Grad Displacement

        if(IsElastic)
        {
            FSpatialTransform ParticleDeformGrad = PushConst.m_ParticleDeformationGrad.GetMatrix(ThreadId.x);
            FSpatialTransform ParticleDeformGradChange = mul(ParticleC, ParticleDeformGrad);
            FSpatialTransform NewDeformGrad = ParticleDeformGrad + ParticleDeformGradChange;
    
            {
                FSpatialTransform U,S,V;
                Math::SVD(NewDeformGrad, U, S, V);
                FSpatialTransform Sc = Math::ClampDiag(S, 1e-6f, 1e30f);

                if(IsVisco)
                {
                    FScalar J = Math::DiagProduct(Sc,0);
                    PushConst.m_ParticleDebug.StoreMatrix(ThreadId.x*2, Sc);

                    FScalar YelidSurface = exp(1.0f-PushConst.m_ViscoPlasticity);
                    FSpatialTransform Scp = Math::ClampDiag(Sc, 1.0/YelidSurface, YelidSurface);
                    PushConst.m_ParticleDebug.StoreMatrix(ThreadId.x*2+1, Scp);

                    FScalar NewJ = Math::DiagProduct(Scp,0);
#ifdef IFSHADER_MPM_3D
                    Sc = Scp * pow(J/NewJ, 1.0f/3.0f);
#else
                    Sc = Scp * sqrt(J/NewJ);
#endif
                }

                FSpatialTransform NewDeformGradSVD = mul(U, mul(Sc, Math::Transpose(V)));
                NewDeformGrad = NewDeformGradSVD;
            }
            PushConst.m_ParticleDeformationGrad.StoreMatrix(ThreadId.x, NewDeformGrad);
        }

        if(IsLiquid)
        {
            FScalar TraceC = Math::Trace(ParticleC);
            FScalar LiquidDensity = PushConst.m_ParticleLiquidDensity.Load(ThreadId.x);
            LiquidDensity *= (TraceC + 1.0f);
            LiquidDensity = max(LiquidDensity, 0.05f); 
            PushConst.m_ParticleLiquidDensity.Store(ThreadId.x, LiquidDensity);
        }

        FSpatialVector Gravity = ToSpatialVector(PushConst.m_Gravity);
        FSpatialVector GravityDisplacement = Gravity * PushConst.m_DeltaTime * PushConst.m_DeltaTime;
        FSpatialVector OverallDisplacement = ParticleDisplacement + GravityDisplacement;

        PushConst.m_ParticleLocation.Store(ThreadId.x, NewParticleLocation);
        PushConst.m_ParticleVelocity.Store(ThreadId.x, OverallDisplacement);
        
    }

}}}
