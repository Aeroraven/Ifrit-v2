#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMBase.h"

namespace Ifrit::Runtime::Artemis
{
    enum class EMPMParticleEmitterTriggerType : u8
    {
        Immediate = 0,
        Periodic  = 1
    };

    enum class EMPMSampleSource : u8
    {
        Random,
        VolumeAsset
    };

    struct MPMParticleEmitterPrivateData;
    class IFRIT_RUNTIME_API IF_CLASS() MPMParticleEmitter : public Component
    {
    public:
        IF_PROPERTY(Editable, UISelect)
        EMPMParticleEmitterTriggerType mEmitTriggerType = EMPMParticleEmitterTriggerType::Periodic;

        IF_PROPERTY(Editable, UISelect)
        MPMSimulatorParticleType mEmitMaterialType = MPMSimulatorParticleType::Fluid;

        IF_PROPERTY(Editable, UISelect)
        EMPMSampleSource mEmitSampleSource = EMPMSampleSource::Random;

        IF_PROPERTY(Editable, UIColor)
        Vector4f mEmitColor = Vector4f(0.0f, 1.0f, 1.0f, 1.0f);

        IF_PROPERTY(Editable, UIText)
        Vector3f mEmitMinRange = Vector3f(0.45f, 0.92f, 0.45f);

        IF_PROPERTY(Editable, UIText)
        Vector3f mEmitMaxRange = Vector3f(0.55f, 0.95f, 0.55f);

        IF_PROPERTY(Editable, UISlider = (min = 5, max = 1000))
        i32 mEmitInterval = 15;

        IF_PROPERTY(Editable, UISlider = (min = 0.001, max = 5.0))
        f32 mParticleMass = 0.5f / 64.0f;

        IF_PROPERTY(Editable, UISlider = (min = 1, max = 20))
        i32 mSamplerPpc = 8;

        IF_PROPERTY(Editable, AssetCategory = "VolumetricData")
        AssetReferenceId mVdbSampleSource;

    private:
        u32                            m_PlaceHolder;
        MPMParticleEmitterPrivateData* m_Data = nullptr;

    public:
        MPMParticleEmitter();
        MPMParticleEmitter(GameObject* owner);
        virtual ~MPMParticleEmitter();

        MPMParticleEmitArgs GetEmitArgs();
        Vec<Vector2f>       GetEmitParticlePosition2D();
        Vec<Vector3f>       GetEmitParticlePosition3D();
        bool                ShouldEmitParticle(i32 frameIdx) const;

    public:
        void IF_FUNCTION()
        ImmediateEmit();
    };

} // namespace Ifrit::Runtime::Artemis
