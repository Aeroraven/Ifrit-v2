#ifndef IFRIT_DLL
    #define IFRIT_DLL
#endif

#define WINDOW_WIDTH 1500
#define WINDOW_HEIGHT 800

#include "ifrit/core/hal/HalDisplay.h"
#include "ifrit/core/hal/HalWindow.h"
#include "ifrit/core/hal/HalHostConcurrency.h"
#include "ifrit/display/presentation/window/WindowSelector.h"

#include "ifrit/vkrhi2/adapter/Device.h"
#include "ifrit/vkrhi2/adapter/Queue.h"
#include "ifrit/vkrhi2/adapter/Backend.h"
#include "ifrit/vkrhi2/adapter/DisplayViewport.h"
#include "ifrit/core/algo/BuddyAllocator.h"
#include "ifrit/rhi/common/RhiShaderResource.h"
#include "ifrit/rhi/common/RhiDynamicUtils.h"
#include "ifrit/shadercompile/helper/ShaderCompileHelper.h"

#include "ifrit/runtime/rendercore/rendergraph/RenderGraph.h"

#include <windows.h>
#include <iostream>

namespace Ifrit
{
    void testShaderRefl()
    {
        HAL::SetCurrentThreadId(0);
        ShaderCompile::ShaderCompileHelper helper;
        helper.SetIncludeBase("D:/Project2/Ifrit-v2/include");
        helper.SetCacheDir("D:/Project2/Ifrit-v2/.cache");
        auto output = helper.CompileShaderFromFile(
            "D:/Project2/Ifrit-v2/include/ifrit.shader.neo/Ayanami/Ayanami.Debug.GlobalDFRayMarch.comp.slang",
            "DebugGlobalDFRayMarchCS", {}, ShaderCompile::EShaderIRFormat::SpirV);
    }

    void testAllocator()
    {
        BuddyAddressAllocator allocator(4, 1024);

        auto                  res = allocator.Allocate(20);
        std::cout << "Allocated offset: " << res.mOffset << ", success: " << res.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;
        // allocator.Free(res.mOffset);
        // std::cout << "Freed offset: " << res.mOffset << std::endl;

        auto res2 = allocator.Allocate(16);
        std::cout << "Allocated offset: " << res2.mOffset << ", success: " << res2.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;
        // allocator.Free(res.mOffset);
        // std::cout << "Freed offset: " << res.mOffset << std::endl;

        auto res3 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res3.mOffset << ", success: " << res3.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res4 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res4.mOffset << ", success: " << res4.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res5 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res5.mOffset << ", success: " << res5.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res6 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res6.mOffset << ", success: " << res6.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res7 = allocator.Allocate(8);
        std::cout << "Allocated offset: " << res7.mOffset << ", success: " << res7.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;

        auto res8 = allocator.Allocate(1024);
        std::cout << "Allocated offset: " << res8.mOffset << ", success: " << res8.mSuccess << std::endl;
        allocator.DumpFreeBlocks();
        std::cout << "----" << std::endl;
        return;
    }

    void testRDG()
    {
        auto backend = RHI::GetRhiBackend();

        using namespace Ifrit::Runtime::RenderCore::RDG;
        RDGGraphBuilder    builder;

        RHI::RhiBufferDesc bufDesc = {};
        bufDesc.mSize              = 114;
        bufDesc.mFlags = RHI::ERhiBufferUsageFlag::UnorderedAccess | RHI::ERhiBufferUsageFlag::StructuredBuffer
            | RHI::ERhiBufferUsageFlag::CopySrc;
        bufDesc.mName            = "TestBuffer";
        auto           buffer    = backend->CreateBuffer(bufDesc);
        auto           bufferUAV = backend->CreateUAV(buffer.Get());

        RDGTextureDesc texDescA = RDGTextureDesc::CreateTexture2D(32, 32, RHI::ERhiImageFormat::R32_SFLOAT);
        auto           texA     = builder.DeclareTexture("TexA", texDescA);
        auto           texB     = builder.DeclareTexture("TexB", texDescA);
        auto           texC     = builder.DeclareTexture("TexC", texDescA);
        auto           texD     = builder.DeclareTexture("TexD", texDescA);
        auto           bufA     = builder.ImportBuffer(buffer);

        struct PassAData
        {
            IRDGAccess_ResourceView* mSRV1;
            IRDGAccess_ResourceView* mSRV2;
            IRDGAccess_ResourceView* mUAV1;
        };

        struct PassBData
        {
            IRDGAccess_ResourceView* mUAV1;
            IRDGAccess_ResourceView* mUAV2;
        };

        struct PassCData
        {
            IRDGAccess_ResourceView* mUAV1;
            IRDGAccess_ResourceView* mUAV2;
        };

        builder.AddPass<PassBData>(
            "PassB", ERDGPassType::Graphics,
            [&](PassBData& data, IRDGGraphBuilderSetupContext& ctx) {
                data.mUAV1 = ctx.CreateUAV(texA, NullOpt, Runtime::RenderCore::RDG::ERDGReadWriteModeFlag::ReadWrite);
                data.mUAV2 = ctx.CreateUAV(texD, NullOpt, Runtime::RenderCore::RDG::ERDGReadWriteModeFlag::ReadWrite);
            },
            [](const PassBData& data, RDGGraphBuilderExecuteContext& ctx) {
                auto uav1Desc = data.mUAV1->GetDescriptor();
            });

        builder.AddPass<PassAData>(
            "PassA", ERDGPassType::AsyncCompute,
            [&](PassAData& data, IRDGGraphBuilderSetupContext& ctx) {
                data.mSRV1 = ctx.CreateSRV(texA, NullOpt);
                data.mSRV2 = ctx.CreateSRV(texB, NullOpt);
                data.mUAV1 = ctx.CreateUAV(bufA, Runtime::RenderCore::RDG::ERDGReadWriteModeFlag::Write);
            },
            [](const PassAData& data, RDGGraphBuilderExecuteContext& ctx) {
                auto srv1Desc = data.mSRV1->GetDescriptor();
                auto srv2Desc = data.mSRV2->GetDescriptor();
                auto uav1Desc = data.mUAV1->GetDescriptor();
            });

        builder.AddPass<PassCData>(
            "PassC", ERDGPassType::AsyncCompute,
            [&](PassCData& data, IRDGGraphBuilderSetupContext& ctx) {
                data.mUAV1 = ctx.CreateUAV(texC, NullOpt, Runtime::RenderCore::RDG::ERDGReadWriteModeFlag::ReadWrite);
                data.mUAV2 = ctx.CreateUAV(bufA, Runtime::RenderCore::RDG::ERDGReadWriteModeFlag::ReadWrite);
            },
            [](const PassCData& data, RDGGraphBuilderExecuteContext& ctx) {
                auto uav1Desc = data.mUAV1->GetDescriptor();
                auto uav2Desc = data.mUAV2->GetDescriptor();
            });

        builder.Compile();
        builder.DumpDebugFile(
            "D:/Project2/Ifrit-v2/rendergraph.dot", ERDGDebugVisualizationMode::PhysicalResourceAlloc);
    }

    void testRhi2()
    {
        using namespace Ifrit;

        Owner<Display::Window::WindowProvider> provider;

        auto                                   dpiScaler = 1.0f;
        dpiScaler                                        = Ifrit::HAL::GetDisplayScale();

        // Setup Window
        Display::Window::WindowProviderSetupArgs winArgs;
        winArgs.useVulkan = true;
        Display::Window::WindowSelector     selector;
        Display::Window::WindowProviderType providerType;
        providerType = Display::Window::WindowProviderType::GLFW;

        provider = selector.CreateWindowProvider(providerType, winArgs);
        provider->Setup(static_cast<usize>(WINDOW_WIDTH), static_cast<usize>(WINDOW_HEIGHT));

        RHI::RhiInitializeArguments initArgs;
        initArgs.mSurfaceWidth      = WINDOW_WIDTH;
        initArgs.mSurfaceHeight     = WINDOW_HEIGHT;
        initArgs.mWin32.m_hInstance = GetModuleHandle(NULL);
        initArgs.mWin32.m_hWnd      = (HWND)provider->GetWindowObject();
        initArgs.mExtensionGetter   = [provider = provider.get()](u32* count) -> const char** {
            return provider->GetVkRequiredInstanceExtensions(count);
        };
        auto                   p         = new RHI::VulkanRHI2::VA_Backend();
        Owner<RHI::RhiBackend> vkBackend = MakeOwner<RHI::VulkanRHI2::VA_Backend>();
        RHI::SetRhiBackend(std::move(vkBackend));

        auto backend = RHI::GetRhiBackend();
        backend->Init(initArgs);

        auto taskScheduler = Task::GetTaskScheduler();

        auto task = taskScheduler->EnqueueTask(
            [&](Task::Task*, void*) {
                auto rhiShaderDesc        = RHI::RhiShaderCreateDesc();
                rhiShaderDesc.mEntryPoint = "TestCS";
                rhiShaderDesc.mSourceType = RHI::ERhiShaderSourceType::SlangCode;
                rhiShaderDesc.mStage      = RHI::ERhiShaderStage::Compute;
                rhiShaderDesc.mFilePath   = "D:/Project2/Ifrit-v2/include/ifrit.shader.neo/TestCS.comp.slang";
                rhiShaderDesc.mName       = "TestCS";

                backend->CreateShader(rhiShaderDesc);
            },
            Task::ENamedTaskThread::AnyThread, {}, nullptr);
        taskScheduler->WaitForTask(task);

        Ifrit::testRDG();

        RHI::RhiBufferDesc bufDesc = {};
        bufDesc.mSize              = 114;
        bufDesc.mFlags = RHI::ERhiBufferUsageFlag::UnorderedAccess | RHI::ERhiBufferUsageFlag::StructuredBuffer
            | RHI::ERhiBufferUsageFlag::CopySrc;
        bufDesc.mName = "TestBuffer";
        auto buffer   = backend->CreateBuffer(bufDesc);

        auto bufferUAV = backend->CreateUAV(buffer.Get());

        provider->Loop([&](int* unused) {
            backend->BeginFrame();

            auto                             variant = backend->GetShaderVariant("TestCS", {});
            RHI::RhiComputePipelineStateDesc compDesc;
            compDesc.mComputeShader = variant;

            auto cmd = RHI::GetCommandListExecutor()->GetImmediateCmdList();

            cmd->SetComputePipelineState(compDesc);

            RHI::RhiShaderParameter params;
            params.SetValue("mData", bufferUAV->GetHandle());
            cmd->SetShaderParameters(params);

            cmd->Dispatch(1, 1, 1);

            backend->EndFrame();
        });

        IF_LOG_INFO("IfritDemo", "RHI Backend released");
    }

} // namespace Ifrit

int main()
{
    // Ifrit::testShaderRefl();
    Ifrit::testRhi2();

    Ifrit::RHI::SetRhiBackend(nullptr);

    return 0;
}