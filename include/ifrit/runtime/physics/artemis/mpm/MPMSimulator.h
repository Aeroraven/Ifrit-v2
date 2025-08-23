#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/rendercore/framegraph/FrameGraphUtils.h"
#include "ifrit/runtime/physics/artemis/mpm/MPMBase.h"
#include "ifrit/runtime/physics/artemis/ArtemisIntegrator.h"
#include "ifrit/core/math/VectorGenerics.h"
#include "ifrit/runtime/base/Scene.h"

namespace Ifrit::Runtime::Artemis
{
    struct MPMSimulatorPrivateData;

    class IFRIT_RUNTIME_API MPMSimulator : public IArtemisSolver
    {
    public:
        MPMSimulator();
        ~MPMSimulator();

        void                SetConfig(const MPMSimulatorConfig& cfg);
        MPMSimulatorConfig& GetActiveConfig();

        virtual void        RunSolverStep(FrameGraphBuilder& builder, f32 deltaTime) override;
        void                CollectScene(Scene* scene) override;
        void                Render(FrameGraphBuilder& builder, FGTextureNode* renderTarget);

        template <u32 Dimension IF_REQUIRES(Dimension == 2 || Dimension == 3)>
        void SetInitParticleLocations(const Vec<TGenericVector<f32, Dimension>>& locations);

        template <u32 Dimension IF_REQUIRES(Dimension == 2 || Dimension == 3)>
        void EmitParticles(const Vec<TGenericVector<f32, Dimension>>& locations, const MPMParticleEmitArgs& args);

        RHI::RhiBufferRef GetParticlePositionBuffer();
        RHI::RhiBufferRef GetParticleCounterBuffer();

        void              RequestClearParticles();
        void              SetDefaultSize(f32 size);

        void              SetDebugRenderTarget(RHI::RhiTexture* rt);
        void              SetMousePosition(f32 x, f32 y);
        void              SetMouseVelocity(f32 vx, f32 vy);
        void              SetMousePushMode(bool enabled);
        void              SetMouseRadAct(f32 activation, f32 radius);

        void              SetMouseDirection(const Vector3f& dir, const Vector3f& camPos);

    private:
        MPMSimulatorPrivateData* m_Data = nullptr;
        MPMSimulatorConfig       m_Config;
    };
} // namespace Ifrit::Runtime::Artemis