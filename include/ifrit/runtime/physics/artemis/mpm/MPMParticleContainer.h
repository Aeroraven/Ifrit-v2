#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMBase.h"
#include "ifrit/runtime/physics/artemis/forwarding/FwdArtemis.h"

namespace Ifrit::Runtime::Artemis
{
    struct MPMParticleContainerProperty
    {
        u32 m_MaxParticleCount = 114514;

        IFRIT_STRUCT_SERIALIZE(m_MaxParticleCount);
    };

    struct MPMParticleContainerPrivateData;

    class IFRIT_RUNTIME_API MPMParticleContainer : public Component, public AttributeOwner<MPMParticleContainerProperty>
    {
    private:
        MPMParticleContainerPrivateData* m_Data              = nullptr;
        bool                             m_IsDeviceDataReady = false;

    public:
        MPMParticleContainer();
        MPMParticleContainer(GameObject* owner);
        virtual ~MPMParticleContainer();

        void   SetupProperties() override;

        IFRIT_COMPONENT_SERIALIZE(m_attributes);

    private:
        MPMGpuParticleBufferCollection* GetDeviceData();
        bool                            GetIsDeviceDataReady() const;
        void                            SetIsDeviceDataReady(bool val);

        friend class MPMSimulator;
    };

} // namespace Ifrit::Runtime::Artemis

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Artemis::MPMParticleContainer)
