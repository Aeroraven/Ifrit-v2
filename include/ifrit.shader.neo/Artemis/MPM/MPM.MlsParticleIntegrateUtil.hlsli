#pragma once

#include "ifrit.shader.neo/Common.hlsli"
#include "ifrit.shader.neo/Artemis/MPM/MPM.Common.hlsli"

namespace IfritShader {
namespace Artemis {
namespace MPM {

    void PerformMlsParticleDeformUpdate(
        int ParticleIndex,
        FScalar DeltaTime,
        FSpatialTransform ParticleC,
        // RWData
        FPackedMatrixHandle hParticleDeformGrad,
        TRWStructuredBufferHandle<FMpmParticleMaterial> hParticleMaterial,
        FRWScalarHandle hParticlePlasticity,
        FPackedMatrixHandle hParticleStressContrib
    )
    {
        FMpmParticleMaterial ParticleMaterial = hParticleMaterial.Load(ParticleIndex);

        bool IsFluid = ParticleMaterial.m_Type == kMpmMaterial_Fluid;
        bool IsSnow = ParticleMaterial.m_Type == kMpmMaterial_Snow;
        bool IsJelly = ParticleMaterial.m_Type == kMpmMaterial_Jelly;
        bool IsVisco = ParticleMaterial.m_Type == kMpmMaterial_Visco;

        FSpatialTransform ParticleDeformGrad = hParticleDeformGrad.GetMatrix(ParticleIndex);
        FSpatialTransform DeformGradChange = DeltaTime * mul(ParticleC, ParticleDeformGrad);
        ParticleDeformGrad += DeformGradChange;

        FScalar TraceC = Math::Trace(ParticleC);
        FScalar Jp = hParticlePlasticity.Load(ParticleIndex);
        FScalar Hardening = exp(10 * (1.0 - Jp));  // From MPM-99

        if(IsJelly) Hardening = 0.3f;

        FSpatialTransform U,S,V;
        Math::SVD(ParticleDeformGrad, U, S, V);
        S = Math::ClampDiag(S, 1e-6f, 1e30f);
        FSpatialTransform Sx = S;

        if(IsSnow)
        {
            Sx = Math::ClampDiag(Sx, 1-2.5e-2f, 1+4.5e-3f);
            Jp *= Math::DiagProductRelative(S,Sx);
        }
        if(IsVisco)
        {
            FScalar YeildSurface = exp(1.0f-0.9f);
            FScalar J1 = Math::DiagProduct(S, 0);
            Sx = Math::ClampDiag(Sx, 1.0f/YeildSurface, YeildSurface);
            FScalar J2 = Math::DiagProduct(Sx, 0);
    #ifdef IFSHADER_MPM_3D
            Sx = Sx * pow(J1/J2, 1.0f/3.0f);
    #else
            Sx = Sx * sqrt(J1/J2);
    #endif
        }
        hParticlePlasticity.Store(ParticleIndex, Jp);
        FScalar J = Math::DiagProduct(Sx,1e-6f);

        if(IsFluid)
        {
    #ifdef IFSHADER_MPM_3D
            ParticleDeformGrad = GetIdentitySpatialTransform() * pow(J, 1.0f/3.0f);
    #else
            ParticleDeformGrad = GetIdentitySpatialTransform() * sqrt(J);
    #endif
        }
        if(IsSnow || IsVisco)
        {
            ParticleDeformGrad = mul(U, mul(Sx, Math::Transpose(V)));
        }
        hParticleDeformGrad.StoreMatrix(ParticleIndex, ParticleDeformGrad);

        FScalar Young = ParticleMaterial.m_Youngs;
        FScalar Possion = ParticleMaterial.m_Possion;
        FScalar Mu, Lambda;
        ToLameParameter( Young, Possion,Mu, Lambda);
        if(IsFluid)
        {
            Mu = 0.0f;
        }
        Mu *= Hardening;
        Lambda *= Hardening;

        FSpatialTransform PFT = 0.0f;
        PFT = FixedCorotatedStressFT(ParticleDeformGrad, J, Mu,Lambda);
        hParticleStressContrib.StoreMatrix(ParticleIndex, PFT);
    }

}}}
