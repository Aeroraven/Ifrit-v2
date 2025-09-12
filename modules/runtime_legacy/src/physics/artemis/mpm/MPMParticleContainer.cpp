#include "ifrit/runtime/physics/artemis/mpm/MPMParticleContainer.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMParticleData.h"

namespace Ifrit::Runtime::Artemis
{
    struct MPMParticleContainerPrivateData
    {
        MPMGpuParticleBufferCollection m_ParticleData;
    };

    IFRIT_APIDECL MPMParticleContainer::MPMParticleContainer()
        : Component(), m_Data(new MPMParticleContainerPrivateData())
    {
    }

    IFRIT_APIDECL MPMParticleContainer::MPMParticleContainer(GameObject* owner)
        : Component(owner), m_Data(new MPMParticleContainerPrivateData())
    {
    }

    IFRIT_APIDECL MPMParticleContainer::~MPMParticleContainer()
    {
        delete m_Data;
        m_Data = nullptr;
    }


    IFRIT_APIDECL MPMGpuParticleBufferCollection* MPMParticleContainer::GetDeviceData()
    {
        return &m_Data->m_ParticleData;
    }
    IFRIT_APIDECL bool MPMParticleContainer::GetIsDeviceDataReady() const { return m_IsDeviceDataReady; }
    IFRIT_APIDECL void MPMParticleContainer::SetIsDeviceDataReady(bool val) { m_IsDeviceDataReady = val; }

} // namespace Ifrit::Runtime::Artemis
