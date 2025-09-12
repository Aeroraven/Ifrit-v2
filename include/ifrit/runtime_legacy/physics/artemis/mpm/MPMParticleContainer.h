#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMBase.h"
#include "ifrit/runtime/physics/artemis/forwarding/FwdArtemis.h"

namespace Ifrit::Runtime::Artemis
{

    struct MPMParticleContainerPrivateData;

    class IFRIT_RUNTIME_API IF_CLASS() MPMParticleContainer : public Component
    {
    public:
        IF_PROPERTY()
        u32 mMaxParticleCount = 114514;

    private:
        MPMParticleContainerPrivateData* m_Data              = nullptr;
        bool                             m_IsDeviceDataReady = false;

    public:
        MPMParticleContainer();
        MPMParticleContainer(GameObject* owner);
        virtual ~MPMParticleContainer();

    private:
        MPMGpuParticleBufferCollection* GetDeviceData();
        bool                            GetIsDeviceDataReady() const;
        void                            SetIsDeviceDataReady(bool val);

        friend class MPMSimulator;
    };

} // namespace Ifrit::Runtime::Artemis
