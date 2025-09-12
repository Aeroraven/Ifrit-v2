#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMSimulator.h"

namespace Ifrit::Runtime::Artemis
{

    struct MPMSimulatorConfiguratorPrivateData;
    class IFRIT_RUNTIME_API IF_CLASS() MPMSimulatorConfigurator : public Component
    {
    public:
        IF_PROPERTY(Editable, UISlider = (min = 1, max = 10))
        i32 mMpmSubsteps = 3;

        IF_PROPERTY(Editable, UISlider = (min = 1, max = 100))
        i32 mPbmpmIterations = 8;

        IF_PROPERTY(Editable, UISlider = (min = 0.0, max = 1.0))
        f32 mPbmpmElasticityRatio = 0.5f;

        IF_PROPERTY(Editable, UISlider = (min = 0.0, max = 5.0))
        f32 mPbmpmElasticityRelax = 1.5f;

        IF_PROPERTY(Editable, UIText)
        Vector3f mGravity = Vector3f(0.0f, -1.0f, 0.0f);

        bool     mClearParticles = false;

        IF_PROPERTY(Editable, UISlider = (min = 0.1, max = 5.0))
        f32 mPointSize = 2.0f;

        IF_PROPERTY(Editable, UISelect)
        bool mEnableRigidCoupling = true;

        IF_PROPERTY(Editable, UISelect)
        bool mEnableRendering = true;

    private:
        u32                                  m_PlaceHolder;
        MPMSimulatorConfiguratorPrivateData* m_Data = nullptr;

    public:
        MPMSimulatorConfigurator();
        MPMSimulatorConfigurator(GameObject* owner);
        virtual ~MPMSimulatorConfigurator();

        void SetActiveSimulator(MPMSimulator* sim);
        void OnUpdate() override;

    public:
        void IF_FUNCTION()
        ClearScene();
    };

} // namespace Ifrit::Runtime::Artemis
