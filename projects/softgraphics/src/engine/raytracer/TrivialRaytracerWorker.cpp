
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

#include "ifrit/softgraphics/engine/raytracer/TrivialRaytracerWorker.h"
#include "ifrit/common/math/VectorOps.h"
#include "ifrit/common/math/simd/SimdVectors.h"
#include "ifrit/softgraphics/engine/raytracer/accelstruct/RtBoundingVolumeHierarchy.h"

using namespace Ifrit::Math::SIMD;

namespace Ifrit::Graphics::SoftGraphics::Raytracer
{
    TrivialRaytracerWorker::TrivialRaytracerWorker(
        std::shared_ptr<TrivialRaytracer>        renderer,
        std::shared_ptr<TrivialRaytracerContext> context, int workerId)
    {
        this->renderer = renderer.get();
        this->context  = context;
        this->workerId = workerId;
    }
    void TrivialRaytracerWorker::Run()
    {
        while (true)
        {
            const auto st = status.load();
            if (st == TrivialRaytracerWorkerStatus::IDLE || st == TrivialRaytracerWorkerStatus::COMPLETED)
            {
                std::this_thread::yield();
            }
            else if (st == TrivialRaytracerWorkerStatus::TERMINATED)
            {
                return;
            }
            else if (st == TrivialRaytracerWorkerStatus::TRACING)
            {
                tracingProcess();
                status.store(TrivialRaytracerWorkerStatus::TRACING_SYNC,
                    std::memory_order::relaxed);
            }
        }
    }
    void TrivialRaytracerWorker::threadCreate()
    {
        thread = std::make_unique<std::thread>(&TrivialRaytracerWorker::Run, this);
    }

    void TrivialRaytracerWorker::tracingProcess()
    {
        // Tracing process

        auto curTile      = 0;
        auto rendererTemp = renderer;
        while ((curTile = rendererTemp->fetchUnresolvedTiles()) >= 0)
        {
            auto tileX = curTile % context->numTileX;
            auto tileY = (curTile / context->numTileX) % context->numTileY;
            auto tileZ = curTile / (context->numTileX * context->numTileY);

            for (int i = 0; i < context->tileWidth; i++)
            {
                if (tileX * context->tileWidth + i >= context->traceRegion.x)
                    break;
                for (int j = 0; j < context->tileHeight; j++)
                {
                    if (tileY * context->tileHeight + j >= context->traceRegion.y)
                        break;
                    for (int k = 0; k < context->tileDepth; k++)
                    {
                        if (tileZ * context->tileDepth + k >= context->traceRegion.z)
                            break;
                        Vector3i invocation = Vector3i(tileX * context->tileWidth + i,
                            tileY * context->tileHeight + j,
                            tileZ * context->tileDepth + k);
                        context->perWorkerRaygen[workerId]->execute(
                            invocation, context->traceRegion, this);
                    }
                }
            }
        }
    }
    void TrivialRaytracerWorker::tracingRecursiveProcess(const RayInternal& ray,
        void* payload, int depth,
        float tmin, float tmax)
    {
        using namespace Ifrit::Math;
        if (depth >= context->maxDepth)
            return;
        auto collresult =
            context->accelerationStructure->queryIntersection(ray, tmin, tmax);
        recurDepth++;
        if (collresult.id == -1)
        {
            if (context->missShader)
            {
                context->perWorkerMiss[workerId]->pushStack(ray, collresult, payload);
                context->perWorkerMiss[workerId]->execute(this);
                context->perWorkerMiss[workerId]->popStack();
            }
        }
        else
        {
            context->perWorkerRayhit[workerId]->pushStack(ray, collresult, payload);
            context->perWorkerRayhit[workerId]->execute(collresult, ray, this);
            context->perWorkerRayhit[workerId]->popStack();
        }
        recurDepth--;
    }
    int TrivialRaytracerWorker::getTracingDepth()
    {
        return recurDepth;
    }
} // namespace Ifrit::Graphics::SoftGraphics::Raytracer
