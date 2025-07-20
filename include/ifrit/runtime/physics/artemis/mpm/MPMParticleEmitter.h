#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMBase.h"

namespace Ifrit::Runtime::Artemis
{

    struct MPMParticleEmitterPrivateData;
    class IFRIT_RUNTIME_API MPMParticleEmitter : public Component
    {
    private:
        u32                            m_PlaceHolder;
        MPMParticleEmitterPrivateData* m_Data = nullptr;

    public:
        MPMParticleEmitter();
        MPMParticleEmitter(GameObject* owner);
        virtual ~MPMParticleEmitter();

        inline String       Serialize() override { return ""; }
        inline void         Deserialize() override {}
        void                SetupProperties() override;

        MPMParticleEmitArgs GetEmitArgs();
        Vec<Vector2f>       GetEmitParticlePosition2D();
        bool                ShouldEmitParticle(i32 frameIdx) const;

        IFRIT_COMPONENT_SERIALIZE(m_PlaceHolder);
    };

} // namespace Ifrit::Runtime::Artemis

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Artemis::MPMParticleEmitter)
