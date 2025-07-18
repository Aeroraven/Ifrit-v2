#include "ifrit/runtime/physics/artemis/mpm/MPMParticleEmitter.h"
#include "ifrit/geomproc/sampler/TrivialRandomSampler.h"

namespace Ifrit::Runtime::Artemis
{
    struct MPMParticleEmitterPrivateData
    {
        MPMSimulatorParticleType m_EmitMaterialType = MPMSimulatorParticleType::Fluid;
        Vector4f                 m_EmitColor        = Vector4f(0.0f, 1.0f, 1.0f, 1.0f);
        Vector3f                 m_EmitMinRange     = Vector3f(0.45f, 0.92f, 0.45f);
        Vector3f                 m_EmitMaxRange     = Vector3f(0.55f, 0.95f, 0.55f);
        i32                      m_EmitKeyFrame     = 15;

        f32                      m_ParticleMass                 = 1.0f;
        f32                      m_ParticleDensity              = 1.0f;
        f32                      m_PbmpmParticleElasticityRatio = 0.5f;
        f32                      m_PbmpmParticleElasticityRelax = 1.5f;
        f32                      m_PbmpmParticleViscoFactor     = 0.1f;
        f32                      m_PbmpmParticleLiquidViscosity = 0.01f;
        f32                      m_PbmpmParticleLiquidRelax     = 1.5f;
        f32                      m_MpmYoungsModulus             = 50.0f;
        f32                      m_MpmPoissonRatio              = 0.3f;
    };

    IFRIT_APIDECL MPMParticleEmitter::MPMParticleEmitter() : m_Data(new MPMParticleEmitterPrivateData()) {}

    IFRIT_APIDECL MPMParticleEmitter::MPMParticleEmitter(Ref<GameObject> owner)
        : Component(owner), m_Data(new MPMParticleEmitterPrivateData())
    {
    }
    IFRIT_APIDECL MPMParticleEmitArgs MPMParticleEmitter::GetEmitArgs()
    {
        MPMParticleEmitArgs args;
        args.m_EmitColor    = m_Data->m_EmitColor;
        args.m_MaterialType = m_Data->m_EmitMaterialType;
        return args;
    }

    IFRIT_APIDECL MPMParticleEmitter::~MPMParticleEmitter() { delete m_Data; }

    IFRIT_APIDECL Vec<Vector2f> MPMParticleEmitter::GetEmitParticlePosition2D()
    {
        GeometryProc::Sampler::TrivialRandomSamplerArgs<f32, 2> args;
        args.m_SampleCount = 100;
        args.m_MinBound    = Vector2f(m_Data->m_EmitMinRange.x, m_Data->m_EmitMinRange.y);
        args.m_MaxBound    = Vector2f(m_Data->m_EmitMaxRange.x, m_Data->m_EmitMaxRange.y);

        auto samples = GeometryProc::Sampler::TrivialRandomSample(args, [](const Vector2f& pos) { return true; });
        return samples;
    }

    IFRIT_APIDECL bool MPMParticleEmitter::ShouldEmitParticle(i32 frameIdx) const
    {
        return (frameIdx % m_Data->m_EmitKeyFrame == 0);
    }

    IFRIT_APIDECL void MPMParticleEmitter::SetupProperties()
    {
        AddEnumProperty<MPMSimulatorParticleType>("Emit Material Type", m_Data->m_EmitMaterialType,
            { MPMSimulatorParticleType::Fluid, MPMSimulatorParticleType::Jelly, MPMSimulatorParticleType::Visco });

        AddProperty<Vector4f, EPropertyEditorType::Color>("Emit Color", m_Data->m_EmitColor);
        AddProperty<Vector3f, EPropertyEditorType::Text>("Emit MinRange", m_Data->m_EmitMinRange);
        AddProperty<Vector3f, EPropertyEditorType::Text>("Emit MaxRange", m_Data->m_EmitMaxRange);
        AddProperty<i32, EPropertyEditorType::Range>("Emit Key Frame", m_Data->m_EmitKeyFrame, 1, 1000);
        AddProperty<f32, EPropertyEditorType::Range>("Particle Mass", m_Data->m_ParticleMass, 0.01f, 100.0f);
        AddProperty<f32, EPropertyEditorType::Range>("Particle Density", m_Data->m_ParticleDensity, 0.01f, 1000.0f);
        AddProperty<f32, EPropertyEditorType::Range>(
            "Elasticity Ratio", m_Data->m_PbmpmParticleElasticityRatio, 0.0f, 1.0f);
        AddProperty<f32, EPropertyEditorType::Range>(
            "Elasticity Relax", m_Data->m_PbmpmParticleElasticityRelax, 1.0f, 10.0f);
        AddProperty<f32, EPropertyEditorType::Range>("Visco Factor", m_Data->m_PbmpmParticleViscoFactor, 0.01f, 1.0f);
        AddProperty<f32, EPropertyEditorType::Range>(
            "Liquid Viscosity", m_Data->m_PbmpmParticleLiquidViscosity, 0.01f, 1.0f);
        AddProperty<f32, EPropertyEditorType::Range>("Liquid Relax", m_Data->m_PbmpmParticleLiquidRelax, 1.0f, 10.0f);
        AddProperty<f32, EPropertyEditorType::Range>("Young's Modulus", m_Data->m_MpmYoungsModulus, 1.0f, 10000.0f);
        AddProperty<f32, EPropertyEditorType::Range>("Poisson Ratio", m_Data->m_MpmPoissonRatio, 0.01f, 0.49f);
    }
} // namespace Ifrit::Runtime::Artemis
