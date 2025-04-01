
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

#include "ifrit/softgraphics/engine/tileraster/TileRasterRenderer.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/softgraphics/engine/tileraster/TileRasterWorker.h"

using namespace Ifrit;

namespace Ifrit::Graphics::SoftGraphics::TileRaster::Inline
{
    template <class T, class U> auto ceilDiv(T a, U b) { return (a + b - 1) / b; }
} // namespace Ifrit::Graphics::SoftGraphics::TileRaster::Inline

namespace Ifrit::Graphics::SoftGraphics::TileRaster
{
    IFRIT_APIDECL TileRasterRenderer::TileRasterRenderer() {}
    IFRIT_APIDECL TileRasterRenderer::~TileRasterRenderer()
    {
        if (!initialized)
        {
            return;
        }
        initialized = false;
        for (auto& worker : workers)
        {
            worker->status.store(TileRasterStage::TERMINATING, std::memory_order::relaxed);
        }
        for (auto& worker : workers)
        {
            worker->execWorker->join();
        }
        ifritLog1("TileRasterRenderer Terminated");
    }
    IFRIT_APIDECL void TileRasterRenderer::bindFrameBuffer(FrameBuffer& frameBuffer)
    {
        this->context->frameBuffer = &frameBuffer;
        context->numTilesX =
            SizeCast<int>(Inline::ceilDiv(frameBuffer.GetColorAttachment(0)->GetWidth(), context->tileWidth));
        context->numTilesY =
            SizeCast<int>(Inline::ceilDiv(frameBuffer.GetColorAttachment(0)->GetHeight(), context->tileWidth));
        context->frameWidth     = context->frameBuffer->GetWidth();
        context->frameHeight    = context->frameBuffer->GetHeight();
        context->invFrameWidth  = 1.0f / context->frameWidth;
        context->invFrameHeight = 1.0f / context->frameHeight;
        updateVectorCapacity();
    }

    IFRIT_APIDECL void TileRasterRenderer::bindVertexBuffer(const VertexBuffer& vertexBuffer)
    {
        this->context->vertexBuffer = &vertexBuffer;
        varyingBufferDirtyFlag      = true;
    }

    IFRIT_APIDECL void TileRasterRenderer::bindFragmentShader(FragmentShader& fragmentShader)
    {
        this->context->fragmentShader = &fragmentShader;
        if (!fragmentShader.isThreadSafe)
        {
            for (int i = 0; i < context->numThreads + 1; i++)
            {
                context->threadSafeFSOwningSection[i] = fragmentShader.getThreadLocalCopy();
                context->threadSafeFS[i]              = context->threadSafeFSOwningSection[i].get();
            }
        }
        else
        {
            for (int i = 0; i < context->numThreads + 1; i++)
            {
                context->threadSafeFS[i] = &fragmentShader;
            }
        }
    }

    IFRIT_APIDECL void TileRasterRenderer::bindUniformBuffer(int binding, int set, BufferManager::IfritBuffer pBuffer)
    {
        auto  p = pBuffer.manager.lock();
        void* data;
        p->mapBufferMemory(pBuffer, &data);
        this->context->uniformMapping[{ binding, set }] = data;
    }

    IFRIT_APIDECL void TileRasterRenderer::bindIndexBuffer(BufferManager::IfritBuffer indexBuffer)
    {
        auto p = indexBuffer.manager.lock();
        p->mapBufferMemory(indexBuffer, (void**)&this->context->indexBuffer);
    }

    IFRIT_APIDECL void TileRasterRenderer::bindVertexShader(VertexShader& vertexShader)
    {
        this->context->owningVaryingDesc =
            std::make_unique<VaryingDescriptor>(std::move(vertexShader.getVaryingDescriptor()));
        bindVertexShaderLegacy(vertexShader, *this->context->owningVaryingDesc);
    }

    IFRIT_APIDECL void TileRasterRenderer::bindVertexShaderLegacy(
        VertexShader& vertexShader, VaryingDescriptor& varyingDescriptor)
    {
        this->context->vertexShader      = &vertexShader;
        this->context->varyingDescriptor = &varyingDescriptor;
        shaderBindingDirtyFlag           = true;
        varyingBufferDirtyFlag           = true;
        if (!vertexShader.isThreadSafe)
        {
            for (int i = 0; i < context->numThreads + 1; i++)
            {
                context->threadSafeVSOwningSection[i] = vertexShader.getThreadLocalCopy();
                context->threadSafeVS[i]              = context->threadSafeVSOwningSection[i].get();
            }
        }
        else
        {
            for (int i = 0; i < context->numThreads + 1; i++)
            {
                context->threadSafeVS[i] = &vertexShader;
            }
        }
    }

    IFRIT_APIDECL void TileRasterRenderer::intializeRenderContext()
    {
        if (varyingBufferDirtyFlag)
        {
            context->vertexShaderResult = std::make_unique<VertexShaderResult>(
                context->vertexBuffer->getVertexCount(), context->varyingDescriptor->getVaryingCounts());
            shaderBindingDirtyFlag = false;
        }
        if (varyingBufferDirtyFlag)
        {
            context->vertexShaderResult->allocateVaryings(context->varyingDescriptor->getVaryingCounts());
            context->varyingDescriptor->applyVaryingDescriptors(context->vertexShaderResult.get());
            context->vertexShaderResult->setVertexCount(context->vertexBuffer->getVertexCount());
            varyingBufferDirtyFlag = false;
        }
    }

    void TileRasterRenderer::createWorkers()
    {
        workers.resize(context->numThreads);
        for (int i = 0; i < context->numThreads; i++)
        {
            workers[i] = std::make_unique<TileRasterWorker>(i, this, context);
            workers[i]->status.store(TileRasterStage::IDLE, std::memory_order::relaxed);
        }
        selfOwningWorker = std::make_unique<TileRasterWorker>(context->numThreads, this, context);
    }
    void TileRasterRenderer::statusTransitionBarrier3(TileRasterStage waitOn, TileRasterStage proceedTo)
    {
        while (true)
        {
            bool allOnBarrier = true;
            for (auto& worker : workers)
            {
                auto expected = waitOn;
                allOnBarrier  = allOnBarrier && (worker->status.load() == waitOn);
            }
            allOnBarrier = allOnBarrier && (selfOwningWorker->status.load() == waitOn);
            if (allOnBarrier)
                break;
        }
        for (auto& worker : workers)
        {
            worker->status.store(proceedTo, std::memory_order::relaxed);
        }
        selfOwningWorker->status.store(proceedTo, std::memory_order::relaxed);
    }
    void TileRasterRenderer::statusTransitionBarrier2(TileRasterStage waitOn, TileRasterStage proceedTo)
    {
        while (true)
        {
            bool allOnBarrier = true;
            for (auto& worker : workers)
            {
                auto expected = waitOn;
                allOnBarrier  = allOnBarrier
                    && (worker->status.compare_exchange_weak(expected, proceedTo, std::memory_order::acq_rel)
                        || (expected >= proceedTo));
            }
            auto expected = waitOn;
            allOnBarrier  = allOnBarrier
                && (selfOwningWorker->status.compare_exchange_weak(expected, proceedTo, std::memory_order::acq_rel)
                    || (expected >= proceedTo));
            if (allOnBarrier)
                break;
            std::this_thread::yield();
        }
    }

    IFRIT_APIDECL void TileRasterRenderer::SetDepthFunc(IfritCompareOp depthFunc)
    {
        context->depthFuncSaved = depthFunc;
        if (context->optDepthTestEnableII)
        {
            context->depthFunc = depthFunc;
        }
    }
    IFRIT_APIDECL void TileRasterRenderer::setBlendFunc(IfritColorAttachmentBlendState state)
    {
        context->blendState = state;
        const auto& bs      = state;
        if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE)
        {
            context->blendColorCoefs.s = { 1, 0, 0, 0 };
        }
        else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO)
        {
            context->blendColorCoefs.s = { 0, 0, 0, 1 };
        }
        else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA)
        {
            context->blendColorCoefs.s = { 0, 0, 1, 0 };
        }
        else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA)
        {
            context->blendColorCoefs.s = { 0, 1, 0, 0 };
        }
        else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        {
            context->blendColorCoefs.s = { 1, 0, -1, 0 };
        }
        else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
        {
            context->blendColorCoefs.s = { 1, -1, 0, 0 };
        }
        else
        {
            ifritError("Unsupported blend factor");
        }
        // SrcAlpha
        if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE)
        {
            context->blendAlphaCoefs.s = { 1, 0, 0, 0 };
        }
        else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO)
        {
            context->blendAlphaCoefs.s = { 0, 0, 0, 1 };
        }
        else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA)
        {
            context->blendAlphaCoefs.s = { 0, 0, 1, 0 };
        }
        else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA)
        {
            context->blendAlphaCoefs.s = { 0, 1, 0, 0 };
        }
        else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        {
            context->blendAlphaCoefs.s = { 1, 0, -1, 0 };
        }
        else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
        {
            context->blendAlphaCoefs.s = { 1, -1, 0, 0 };
        }
        else
        {
            ifritError("Unsupported blend factor");
        }

        // DstColor
        if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE)
        {
            context->blendColorCoefs.d = { 1, 0, 0, 0 };
        }
        else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO)
        {
            context->blendColorCoefs.d = { 0, 0, 0, 1 };
        }
        else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA)
        {
            context->blendColorCoefs.d = { 0, 0, 1, 0 };
        }
        else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA)
        {
            context->blendColorCoefs.d = { 0, 1, 0, 0 };
        }
        else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        {
            context->blendColorCoefs.d = { 1, 0, -1, 0 };
        }
        else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
        {
            context->blendColorCoefs.d = { 1, -1, 0, 0 };
        }
        else
        {
            ifritError("Unsupported blend factor");
        }

        // DstAlpha
        if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE)
        {
            context->blendAlphaCoefs.d = { 1, 0, 0, 0 };
        }
        else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO)
        {
            context->blendAlphaCoefs.d = { 0, 0, 0, 1 };
        }
        else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA)
        {
            context->blendAlphaCoefs.d = { 0, 0, 1, 0 };
        }
        else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA)
        {
            context->blendAlphaCoefs.d = { 0, 1, 0, 0 };
        }
        else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        {
            context->blendAlphaCoefs.d = { 1, 0, -1, 0 };
        }
        else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA)
        {
            context->blendAlphaCoefs.d = { 1, -1, 0, 0 };
        }
        else
        {
            ifritError("Unsupported blend factor");
        }
    }

    int TileRasterRenderer::fetchUnresolvedTileRaster()
    {
        auto counter = unresolvedTileRaster.fetch_sub(1) - 1;
        return counter;
    }
    int TileRasterRenderer::fetchUnresolvedTileFragmentShading()
    {
        auto counter = unresolvedTileFragmentShading.fetch_sub(1) - 1;
        return counter;
    }
    int TileRasterRenderer::fetchUnresolvedTileSort()
    {
        auto counter = unresolvedTileSort.fetch_sub(1) - 1;
        return counter;
    }
    int TileRasterRenderer::fetchUnresolvedChunkVertex()
    {
        auto counter = unresolvedChunkVertex.fetch_sub(1) - 1;
        return counter;
    }
    int TileRasterRenderer::fetchUnresolvedChunkGeometry()
    {
        auto counter = unresolvedChunkGeometry.fetch_sub(1) - 1;
        return counter;
    }
    IFRIT_APIDECL void TileRasterRenderer::optsetForceDeterministic(bool opt) { context->optForceDeterministic = opt; }
    IFRIT_APIDECL void TileRasterRenderer::optSetDepthTestEnable(bool opt)
    {
        context->optDepthTestEnableII = opt;
        if (!opt)
        {
            context->depthFunc = IF_COMPARE_OP_ALWAYS;
        }
        else
        {
            context->depthFunc = context->depthFuncSaved;
        }
    }
    void TileRasterRenderer::resetWorkers(TileRasterStage expectedStage)
    {
        for (auto& worker : workers)
        {
            worker->status.store(expectedStage, std::memory_order::relaxed);
        }
        // No need to set self-owning worker
        // selfOwningWorker->status.store(expectedStage, std::memory_order::relaxed);
    }
    void TileRasterRenderer::updateVectorCapacity()
    {
        auto totalTiles = context->numTilesX * context->numTilesY;
        context->sortedCoverQueue.resize(totalTiles);
        for (int i = 0; i < context->numThreads + 1; i++)
        {
            context->rasterizerQueue[i].resize(totalTiles);
            context->coverQueue[i].resize(totalTiles);
        }
    }

    void TileRasterRenderer::updateUniformBuffer()
    {
        auto vsUniforms = context->vertexShader->getUniformList();
        auto fsUniforms = context->fragmentShader->getUniformList();
        for (int i = context->numThreads; i >= 0; i--)
        {
            for (const auto& x : vsUniforms)
            {
                if (context->uniformMapping.count(x))
                {
                    context->threadSafeVS[i]->updateUniformData(x.first, x.second, context->uniformMapping[x]);
                }
            }
            for (const auto& x : fsUniforms)
            {
                if (context->uniformMapping.count(x))
                {
                    context->threadSafeFS[i]->updateUniformData(x.first, x.second, context->uniformMapping[x]);
                }
            }
        }
    }

    IFRIT_APIDECL void TileRasterRenderer::Init()
    {
        context                         = std::make_shared<TileRasterContext>();
        context->blendState.blendEnable = false;

        createWorkers();
        for (auto& worker : workers)
        {
            worker->status.store(TileRasterStage::COMPLETED, std::memory_order::relaxed);
            worker->threadStart();
        }
        selfOwningWorker->status.store(TileRasterStage::COMPLETED, std::memory_order::relaxed);
        initialized = true;
    }
    IFRIT_APIDECL void TileRasterRenderer::clear()
    {
        context->frameBuffer->GetColorAttachment(0)->clearImageZero();
        context->frameBuffer->GetDepthAttachment()->clearImage(255);
    }

    IFRIT_APIDECL void TileRasterRenderer::drawElements(int vertexCount, bool clearFramebuffer) IFRIT_AP_NOTHROW
    {
        intializeRenderContext();
        updateUniformBuffer();
        context->indexBufferSize = vertexCount;
        unresolvedTileRaster.store(context->numTilesX * context->numTilesY, std::memory_order::relaxed);
        auto vertexChunks   = Inline::ceilDiv(context->vertexBuffer->getVertexCount(), context->vsChunkSize);
        auto geometryChunks = Inline::ceilDiv(vertexCount / context->vertexStride, context->gsChunkSize);
        unresolvedChunkVertex.store(vertexChunks, std::memory_order::relaxed);
        unresolvedChunkGeometry.store(geometryChunks, std::memory_order::relaxed);
        if (clearFramebuffer)
        {
            resetWorkers(TileRasterStage::DRAWCALL_START_CLEAR);
            selfOwningWorker->drawCall(true);
        }
        else
        {
            resetWorkers(TileRasterStage::DRAWCALL_START);
            selfOwningWorker->drawCall(false);
        }
        statusTransitionBarrier3(TileRasterStage::FRAGMENT_SHADING_SYNC, TileRasterStage::IDLE);
    }

} // namespace Ifrit::Graphics::SoftGraphics::TileRaster