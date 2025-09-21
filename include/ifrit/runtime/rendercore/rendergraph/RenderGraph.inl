#pragma once
#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"

namespace Ifrit::Runtime::RenderCore::RDG
{
    template <IDefaultCopyable TPassData, typename TFnSetup, typename TFnExecute>
        requires IConceptRDGPassFnSetup<TFnSetup, TPassData> && IConceptRDGPassFnExecute<TFnExecute, TPassData>
        && IConceptConvertibleToFuncPtr<TFnExecute>
    RDGPassHandle RDGGraphBuilder::AddPass(const String& name, ERDGPassType type, TFnSetup setup, TFnExecute execute)
    {
        u32         passIdx  = PreAllocatePass();
        RDGPassData passData = RDGPassData::Create<TPassData>();
        auto        setupFn  = [setup = std::move(setup), passIdx, this]() {
            auto& data         = GetPassData(passIdx).As<TPassData>();
            auto& setupContext = GetSetupContext();
            setup(data, setupContext);
        };

        auto executeFn = [executePtr = std::move(execute), passIdx, this](void) {
            const auto& data    = GetPassData(passIdx).As<TPassData>();
            auto&       context = GetExecuteContext();
            executePtr(data, context);
        };
        return AddPassInternal(passIdx, name, std::move(passData), type, std::move(setupFn), std::move(executeFn));
    }
} // namespace Ifrit::Runtime::RenderCore::RDG