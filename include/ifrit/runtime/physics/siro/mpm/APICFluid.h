#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"

namespace Ifrit::Runtime::Siro
{
    struct APICFluidPrivateData;

    class IFRIT_RUNTIME_API APICFluid
    {
    public:
        APICFluid();
        ~APICFluid();

        void RunSolver(FrameGraphBuilder& builder, FGTextureNode* renderTarget);

    private:
        APICFluidPrivateData* m_Data = nullptr;
    };
} // namespace Ifrit::Runtime::Siro