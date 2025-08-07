#include "ifrit/runtime/physics/artemis/mpm/MPMParticleEmitter.h"
#include "ifrit/geomproc/sampler/TrivialRandomSampler.h"
#include "ifrit/runtime/asset/VolumeAsset.h"
namespace Ifrit::Runtime::Artemis
{
    struct MPMParticleEmitterPrivateData
    {
        f32  m_ParticleDensity              = 1.0f;
        f32  m_PbmpmParticleElasticityRatio = 0.5f;
        f32  m_PbmpmParticleElasticityRelax = 1.5f;
        f32  m_PbmpmParticleViscoFactor     = 0.1f;
        f32  m_PbmpmParticleLiquidViscosity = 0.01f;
        f32  m_PbmpmParticleLiquidRelax     = 1.5f;
        f32  m_MpmYoungsModulus             = 50.0f;
        f32  m_MpmPoissonRatio              = 0.3f;

        bool m_ImmediateEmit = false;
    };

    IFRIT_APIDECL MPMParticleEmitter::MPMParticleEmitter() : m_Data(new MPMParticleEmitterPrivateData()) {}

    IFRIT_APIDECL MPMParticleEmitter::MPMParticleEmitter(GameObject* owner)
        : Component(owner), m_Data(new MPMParticleEmitterPrivateData())
    {
    }
    IFRIT_APIDECL MPMParticleEmitArgs MPMParticleEmitter::GetEmitArgs()
    {
        MPMParticleEmitArgs args;
        args.m_EmitColor    = mEmitColor;
        args.m_MaterialType = mEmitMaterialType;
        args.m_Mass         = mParticleMass;
        return args;
    }

    IFRIT_APIDECL MPMParticleEmitter::~MPMParticleEmitter() { delete m_Data; }

    IFRIT_APIDECL Vec<Vector2f> MPMParticleEmitter::GetEmitParticlePosition2D()
    {
        IF_LOG_ASSERTION("MPMParticleEmitter", mEmitSampleSource == EMPMSampleSource::Random,
            "Only Random sample source is supported for 2D particle emission");

        GeometryProc::Sampler::TrivialRandomSamplerArgs<f32, 2> args;
        args.m_SampleCount = 100;
        args.m_MinBound    = Vector2f(mEmitMinRange.x, mEmitMinRange.y);
        args.m_MaxBound    = Vector2f(mEmitMaxRange.x, mEmitMaxRange.y);

        auto samples = GeometryProc::Sampler::TrivialRandomSample(args, [](const Vector2f& pos) { return true; });
        return samples;
    }
    IFRIT_APIDECL Vec<Vector3f> MPMParticleEmitter::GetEmitParticlePosition3D()
    {
        if (mEmitSampleSource == EMPMSampleSource::VolumeAsset)
        {
            auto                         assetRegistry = Runtime::GetActiveApplication()->GetAssetRegistry();
            auto                         asset         = assetRegistry->GetAsset<VolumeAsset>(mVdbSampleSource.mGuid);

            Geometry::VolumeSamplingArgs args;
            args.mDeltaCellX             = 0.3f;
            args.mPPC                    = mSamplerPpc;
            args.mTransform_DoNormalize  = true;
            args.mTransform_MoveToCenter = Vector3f(32.0f, 32.0f, 32.0f);
            args.mTransform_NormMinBound = Vector3f(0.0f, 0.0f, 0.0f);
            args.mTransform_NormMaxBound = Vector3f(64.0f, 64.0f, 64.0f);
            auto samples                 = asset->SampleAsPointCloud(args);
            return samples;
        }
        else
        {
            GeometryProc::Sampler::TrivialRandomSamplerArgs<f32, 3> args;
            args.m_SampleCount = 100;
            args.m_MinBound    = Vector3f(mEmitMinRange.x, mEmitMinRange.y, mEmitMinRange.z);
            args.m_MaxBound    = Vector3f(mEmitMaxRange.x, mEmitMaxRange.y, mEmitMaxRange.z);

            auto samples = GeometryProc::Sampler::TrivialRandomSample(args, [](const Vector3f& pos) { return true; });
            return samples;
        }
    }

    IFRIT_APIDECL bool MPMParticleEmitter::ShouldEmitParticle(i32 frameIdx) const
    {
        if (m_Data->m_ImmediateEmit)
        {
            m_Data->m_ImmediateEmit = false;
            return true;
        }
        if (mEmitTriggerType == EMPMParticleEmitterTriggerType::Periodic)
        {
            return (frameIdx % mEmitInterval == 0);
        }
        return false;
    }

    void MPMParticleEmitter::ImmediateEmit() { m_Data->m_ImmediateEmit = true; }

} // namespace Ifrit::Runtime::Artemis
