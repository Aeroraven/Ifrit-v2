#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"

namespace Ifrit::Runtime::Siro
{
    class TrivialPBDClothPrivateData;

    class IFRIT_RUNTIME_API TrivialPBDCloth
    {
    public:
        TrivialPBDCloth();
        ~TrivialPBDCloth();

        void Initialize(FrameGraphBuilder& builder, u32 width, u32 height, u32 mass);
        void VelocityUpdatePre(FrameGraphBuilder& builder, f32 deltaTime);
        void DampVelocity(FrameGraphBuilder& builder, f32 dampingFactor);
        void InitialOffsetGeneration(FrameGraphBuilder& builder, f32 deltaTime);
        void ProjectConstraintsSingleIteration(FrameGraphBuilder& builder);
        void ProjectConstraints(FrameGraphBuilder& builder, u32 iterations);
        void ApplyAdjustion(FrameGraphBuilder& builder);
        void VelocityUpdatePost(FrameGraphBuilder& builder);

        void Advance(FrameGraphBuilder& builder, f32 deltaTime, u32 solverIters);

    private:
        TrivialPBDClothPrivateData* m_Data = nullptr;
    };
} // namespace Ifrit::Runtime::Siro