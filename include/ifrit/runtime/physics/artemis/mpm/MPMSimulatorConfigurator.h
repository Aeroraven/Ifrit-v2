#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMSimulator.h"

namespace Ifrit::Runtime::Artemis
{

    struct MPMSimulatorConfiguratorPrivateData;
    class IFRIT_RUNTIME_API MPMSimulatorConfigurator : public Component
    {
    private:
        u32                                  m_PlaceHolder;
        MPMSimulatorConfiguratorPrivateData* m_Data = nullptr;

    public:
        MPMSimulatorConfigurator();
        MPMSimulatorConfigurator(GameObject* owner);
        virtual ~MPMSimulatorConfigurator();

        inline String Serialize() override { return ""; }
        inline void   Deserialize() override {}
        void          SetupProperties() override;

        void          SetActiveSimulator(MPMSimulator* sim);
        void          OnUpdate() override;

        IFRIT_COMPONENT_SERIALIZE(m_PlaceHolder);
    };

} // namespace Ifrit::Runtime::Artemis

IFRIT_COMPONENT_REGISTER(Ifrit::Runtime::Artemis::MPMSimulatorConfigurator)
