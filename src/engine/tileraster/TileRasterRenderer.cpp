#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tileraster/TileRasterRenderer.h"


namespace Ifrit::Engine::TileRaster {
	TileRasterRenderer::TileRasterRenderer() {
		
	}
	void TileRasterRenderer::bindFrameBuffer(FrameBuffer& frameBuffer) {
		this->context->frameBuffer = &frameBuffer;
	}
	void TileRasterRenderer::bindVertexBuffer(const VertexBuffer& vertexBuffer) {
		this->context->vertexBuffer = &vertexBuffer;
		varyingBufferDirtyFlag = true;
	}
	void TileRasterRenderer::bindFragmentShader(FragmentShader& fragmentShader) {
		this->context->fragmentShader = &fragmentShader;
	}
	void TileRasterRenderer::bindIndexBuffer(const std::vector<int>& indexBuffer) {
		this->context->indexBuffer = &indexBuffer;
	}
	void TileRasterRenderer::bindVertexShader(VertexShader& vertexShader) {
		this->context->vertexShader = &vertexShader;
		shaderBindingDirtyFlag = true;
		varyingBufferDirtyFlag = true;
	}
	void TileRasterRenderer::intializeRenderContext() {
		if (varyingBufferDirtyFlag) {
			context->vertexShaderResult = std::make_unique<VertexShaderResult>(
			context->vertexBuffer->getVertexCount(), context->vertexShader->getVaryingCounts());
		}
		if (varyingBufferDirtyFlag) {
			context->vertexShaderResult->allocateVaryings(context->vertexShader->getVaryingCounts());
			context->vertexShader->applyVaryingDescriptors(context->vertexShaderResult.get());
			context->vertexShaderResult->setVertexCount(context->vertexBuffer->getVertexCount());
		}
		varyingBufferDirtyFlag = false;
		shaderBindingDirtyFlag = false;

	}
	void TileRasterRenderer::createWorkers() {
		workers.resize(context->numThreads);
		for (int i = 0; i < context->numThreads; i++) {
			workers[i] = std::make_unique<TileRasterWorker>(i, shared_from_this(), context);
			workers[i]->status.store(TileRasterStage::CREATED);
		}
	}
	void TileRasterRenderer::statusTransitionBarrier(TileRasterStage waitOn, TileRasterStage proceedTo) {
		bool flag = false;
		while (!flag) {
			bool allOnBarrier = true;
			for (auto& worker : workers) {
				auto arrived = worker->status.load() == waitOn;
				auto advanced = worker->status.load() >= proceedTo;
				allOnBarrier = allOnBarrier && (arrived || advanced);
			}
			if (allOnBarrier) {
				for (auto& worker : workers){
					worker->status.store(proceedTo);
					worker->activated.store(true);
				}
				flag = true;
			}
			else {
				std::this_thread::yield();

			}
		}
		return ;
	}
	void TileRasterRenderer::waitOnWorkers(TileRasterStage waitOn){
		bool flag = false;
		while (!flag) {
			bool allOnBarrier = true;
			for (auto& worker : workers) {
				auto arrived = worker->status.load() == waitOn;
				allOnBarrier = allOnBarrier && arrived;
			}
			if (allOnBarrier) flag = true;
		}
	}
	int TileRasterRenderer::fetchUnresolvedTileRaster() {
		auto counter = unresolvedTileRaster.fetch_add(1);
		if (counter >= context->tileBlocksX * context->tileBlocksX) {
			return -1;
		}
		return counter;
	}
	int TileRasterRenderer::fetchUnresolvedTileFragmentShading() {
		auto counter = unresolvedTileFragmentShading.fetch_add(1);
		if (counter >= context->tileBlocksX * context->tileBlocksX) {
			return -1;
		}
		return counter;
	}
	void TileRasterRenderer::resetWorkers() {
		for (auto& worker : workers) {
			worker->status.store(TileRasterStage::VERTEX_SHADING);
			worker->activated.store(true);
		}
	}
	void TileRasterRenderer::init() {
		context = std::make_shared<TileRasterContext>();
		context->rasterizerQueue.resize(context->numThreads);
		context->coverQueue.resize(context->numThreads);
		context->workerIdleTime.resize(context->numThreads);
		context->assembledTriangles.resize(context->numThreads);
		for (int i = 0; i < context->numThreads; i++) {
			context->rasterizerQueue[i].resize(context->tileBlocksX * context->tileBlocksX);
			context->coverQueue[i].resize(context->tileBlocksX * context->tileBlocksX);
		}
		createWorkers();
		for (auto& worker : workers) {
			worker->status.store(TileRasterStage::CREATED);
			worker->threadStart();
		}
	}
	void TileRasterRenderer::clear() {
		std::lock_guard lockGuard(lock);
		context->frameBuffer->getColorAttachment(0)->clearImage();
		context->frameBuffer->getDepthAttachment()->clearImage(255);
	}

	void TileRasterRenderer::render() {
		std::lock_guard lockGuard(lock);
		unresolvedTileRaster.store(0,std::memory_order_seq_cst);
		unresolvedTileFragmentShading.store(0, std::memory_order_seq_cst);
		for (int i = 0; i < context->numThreads; i++) {
			context->workerIdleTime[i] = 0;
			context->assembledTriangles[i].clear();
			for (int j = 0; j < context->tileBlocksX * context->tileBlocksX; j++) {
				auto prevSizeRQ = context->rasterizerQueue[i][j].size();
				auto prevSizeCQ = context->coverQueue[i][j].size();
				context->rasterizerQueue[i][j].clear();
				context->coverQueue[i][j].clear();

			}
		}
		context->primitiveMinZ.reserve(context->indexBuffer->size() / context->vertexStride);
		context->primitiveMinZ.resize(context->indexBuffer->size() / context->vertexStride);
		context->primitiveEdgeCoefs.reserve(context->indexBuffer->size() / context->vertexStride);
		context->primitiveEdgeCoefs.resize(context->indexBuffer->size() / context->vertexStride);

		intializeRenderContext();
		resetWorkers();
		statusTransitionBarrier(TileRasterStage::VERTEX_SHADING_SYNC, TileRasterStage::GEOMETRY_PROCESSING);
		statusTransitionBarrier(TileRasterStage::GEOMETRY_PROCESSING_SYNC, TileRasterStage::RASTERIZATION);
		statusTransitionBarrier(TileRasterStage::RASTERIZATION_SYNC, TileRasterStage::FRAGMENT_SHADING);
		statusTransitionBarrier(TileRasterStage::FRAGMENT_SHADING_SYNC, TileRasterStage::TERMINATED);
		waitOnWorkers(TileRasterStage::TERMINATED);
	}

}