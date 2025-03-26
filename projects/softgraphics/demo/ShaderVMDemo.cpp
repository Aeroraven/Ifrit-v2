
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#include "ShaderVMDemo.h"
#include "./demo/shader/DefaultDemoShaders.cuh"
#include "core/data/Image.h"
#include "engine/bufferman/BufferManager.h"
#include "engine/comllvmrt/WrappedLLVMRuntime.h"
#include "engine/shadervm/spirv/SpvVMInterpreter.h"
#include "engine/shadervm/spirv/SpvVMReader.h"
#include "engine/shadervm/spirv/SpvVMShader.h"
#include "engine/shadervm/spirvvec/SpvMdQuadIRGenerator.h"
#include "engine/shadervm/spirvvec/SpvMdShader.h"
#include "engine/tileraster/TileRasterRenderer.h"
#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tilerastercuda/TileRasterCoreInvocationCuda.cuh"
#include "engine/tilerastercuda/TileRasterRendererCuda.h"
#include "ifrit/common/math/LinalgOps.h"
#include "presentation/backend/OpenGLBackend.h"
#include "presentation/backend/TerminalAsciiBackend.h"
#include "presentation/backend/TerminalCharColorBackend.h"
#include "presentation/window/GLFWWindowProvider.h"
#include "utility/loader/ImageLoader.h"
#include "utility/loader/WavefrontLoader.h"

using namespace std;
using namespace Ifrit::SoftRenderer::Core::Data;
using namespace Ifrit::SoftRenderer::TileRaster;
using namespace Ifrit::SoftRenderer::Utility::Loader;
#ifdef IFRIT_FEATURE_CUDA
using namespace Ifrit::SoftRenderer::Math::ShaderOps;
#endif
using namespace Ifrit::Display::Window;
using namespace Ifrit::Display::Backend;
using namespace Ifrit::Math;
using namespace Ifrit::SoftRenderer::ShaderVM::Spirv;
using namespace Ifrit::SoftRenderer::ShaderVM::SpirvVec;
using namespace Ifrit::SoftRenderer::ComLLVMRuntime;
using namespace Ifrit::SoftRenderer::BufferManager;

namespace Ifrit::Demo::ShaderVMDemo
{

    int mainTest2()
    {
        SpvVMReader  reader;
        SpvVMContext ctx;
        reader.initializeContext(&ctx);
        auto cv = reader.readFile(IFRIT_ASSET_PATH "/shaders/diffuse.frag.hlsl.spv");
        reader.parseByteCode(cv.data(), cv.size() / 4, &ctx);

        SpVcQuadGroupedIRGenerator intp;
        SpVcVMGeneratorContext     irctx;
        intp.bindBytecode(&ctx, &irctx);
        intp.Init();

        intp.parse();
        intp.verbose();

        auto                      ir = intp.generateIR();
        WrappedLLVMRuntimeBuilder builder;
        auto                      ax = builder.buildRuntime();
        ax->loadIR(ir, "identifier");
        return 0;
    }

    int mainTest()
    {
        // Matrix4x4f view = (LookAt({ 0,0.01,0.02 }, { 0,0.01,0.0 }, { 0,1,0 }));
        // Matrix4x4f view = (LookAt({ 0,0.75,1.50 }, { 0,0.75,0.0 }, { 0,1,0 }));
        Matrix4x4f            view = (LookAt({ 0, 0.1, 0.25 }, { 0, 0.1, 0.0 }, { 0, 1, 0 }));
        // Matrix4x4f view = (LookAt({ 0,1.95,1.50 }, { 0,0.95,0.0 }, { 0,1,0 }));
        // Matrix4x4f proj = (perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1,
        // 3000)); Matrix4x4f view = (LookAt({ 0,60.0,130.0 }, { 0,60.0,0.0 }, { 0,1,0
        // }));

        // Matrix4x4f view = (LookAt({ 500,300,0 }, { -100,300,-0 }, { 0,1,0 }));
        // Matrix4x4f view = (LookAt({ 0,1.5,0 }, { -100,1.5,0 }, { 0,1,0 }));
        Matrix4x4f            proj = (Perspective(60 * 3.14159 / 180, 1920.0 / 1080.0, 0.1, 3000));

        Matrix4x4f            mvp = Transpose(MatMul(proj, view));

        WavefrontLoader       loader;
        std::vector<Vector3f> pos;
        std::vector<Vector3f> normal;
        std::vector<Vector2f> uv;
        std::vector<uint32_t> index;
        std::vector<Vector3f> procNormal;
        loader.loadObject(IFRIT_ASSET_PATH "/bunny.obj", pos, normal, uv, index);
        procNormal = loader.RemapNormals(normal, index, pos.size());

        IF_CONSTEXPR int                      DEMO_RESOLUTION_X = 2048;
        IF_CONSTEXPR int                      DEMO_RESOLUTION_Y = 2048;
        std::shared_ptr<ImageF32>             image             = std::make_shared<ImageF32>(DEMO_RESOLUTION_X, DEMO_RESOLUTION_Y, 4);
        std::shared_ptr<ImageF32>             depth             = std::make_shared<ImageF32>(DEMO_RESOLUTION_X, DEMO_RESOLUTION_Y, 1);
        std::shared_ptr<TileRasterRenderer>   renderer          = std::make_shared<TileRasterRenderer>();
        std::shared_ptr<TrivialBufferManager> bufferman         = std::make_shared<TrivialBufferManager>();
        bufferman->Init();
        FrameBuffer  frameBuffer;

        VertexBuffer vertexBuffer;
        vertexBuffer.setLayout({ TypeDescriptors.FLOAT4, TypeDescriptors.FLOAT4 });
        vertexBuffer.allocateBuffer(pos.size());
        for (int i = 0; i < pos.size(); i++)
        {
            vertexBuffer.setValue(i, 0, Vector4f(pos[i].x, pos[i].y, pos[i].z, 1));
            vertexBuffer.setValue(i, 1, Vector4f(procNormal[i].x, procNormal[i].y, procNormal[i].z, 0.5f));
        }

        std::vector<int> indexBuffer = { 0, 1, 2, 2, 3, 0 };
        indexBuffer.resize(index.size() / 3);
        for (int i = 0; i < index.size(); i += 3)
        {
            indexBuffer[i / 3] = index[i];
        }
        printf("Num Triangles: %lld\n", indexBuffer.size() / 3);

        frameBuffer.SetColorAttachments({ image.get() });
        frameBuffer.SetDepthAttachment(*depth);

        renderer->Init();
        renderer->bindFrameBuffer(frameBuffer);
        renderer->bindVertexBuffer(vertexBuffer);

        renderer->optsetForceDeterministic(false);
        renderer->optSetDepthTestEnable(true);

        IfritColorAttachmentBlendState blendState;
        blendState.blendEnable         = false;
        blendState.srcColorBlendFactor = IF_BLEND_FACTOR_SRC_ALPHA;
        blendState.dstColorBlendFactor = IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendState.srcAlphaBlendFactor = IF_BLEND_FACTOR_SRC_ALPHA;
        blendState.dstAlphaBlendFactor = IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        renderer->setBlendFunc(blendState);

        struct Uniform
        {
            Vector4f t1 = { 1, 1, 1, 0 };
            Vector4f t2 = { 0.0, 0, 0, 0 };
        } uniform;

        auto uniform1 = bufferman->CreateBuffer({ sizeof(uniform) });
        bufferman->bufferData(uniform1, &uniform, 0, sizeof(uniform));
        auto uniform2 = bufferman->CreateBuffer({ sizeof(mvp) });
        bufferman->bufferData(uniform2, &mvp, 0, sizeof(mvp));
        auto indexBuffer1 = bufferman->CreateBuffer({ sizeof(indexBuffer[0]) * indexBuffer.size() });
        bufferman->bufferData(indexBuffer1, indexBuffer.data(), 0, sizeof(indexBuffer[0]) * indexBuffer.size());

        renderer->bindUniformBuffer(0, 0, uniform1);
        renderer->bindUniformBuffer(1, 0, uniform2);
        renderer->bindIndexBuffer(indexBuffer1);

        SpvVMReader               reader;
        auto                      fsCode = reader.readFile(IFRIT_ASSET_PATH "/shaders/demo.frag.hlsl.spv");
        auto                      vsCode = reader.readFile(IFRIT_ASSET_PATH "/shaders/demo.vert.hlsl.spv");

        WrappedLLVMRuntimeBuilder llvmRuntime;
        SpvVertexShader           vertexShader(llvmRuntime, vsCode);
        renderer->bindVertexShader(vertexShader);
        // SpvVecFragmentShader fragmentShader(llvmRuntime, fsCode);
        SpvFragmentShader fragmentShader(llvmRuntime, fsCode);
        renderer->bindFragmentShader(fragmentShader);

        GLFWWindowProvider windowProvider;
        windowProvider.Setup(1920, 1080);
        windowProvider.SetTitle("Ifrit-v2 CPU Multithreading");

        OpenGLBackend backend;
        backend.SetViewport(0, 0, windowProvider.GetWidth(), windowProvider.GetHeight());
        windowProvider.Loop([&](int* coreTime) {
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            uniform.t1.x =
                0.4f * std::sin((float)std::chrono::duration_cast<std::chrono::milliseconds>(start.time_since_epoch()).count() / 1000.0f);
            bufferman->bufferData(uniform1, &uniform, 0, sizeof(uniform));
            renderer->drawElements(indexBuffer.size(), true);
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            *coreTime                                          = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            backend.UpdateTexture(*image);
            backend.draw();
        });
        return 0;
    }
} // namespace Ifrit::Demo::ShaderVMDemo