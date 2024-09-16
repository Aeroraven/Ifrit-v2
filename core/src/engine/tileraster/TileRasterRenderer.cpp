#include "engine/tileraster/TileRasterWorker.h"
#include "engine/tileraster/TileRasterRenderer.h"

namespace Ifrit::Engine::TileRaster::Inline {
	template<class T,class U>
	auto ceilDiv(T a, U b) {
		return (a + b - 1) / b;
	}
}

namespace Ifrit::Engine::TileRaster {
	IFRIT_APIDECL TileRasterRenderer::TileRasterRenderer() {
		
	}
	IFRIT_APIDECL TileRasterRenderer::~TileRasterRenderer(){
		if (!initialized) {
			return;
		}
		initialized = false;
		for (auto& worker : workers) {
			worker->status.store(TileRasterStage::TERMINATING, std::memory_order::relaxed);
			worker->activated.store(true);
		}
		for (auto& worker : workers) {
			worker->execWorker->join();
		}
		ifritLog1("TileRasterRenderer Terminated");
	}
	IFRIT_APIDECL void TileRasterRenderer::bindFrameBuffer(FrameBuffer& frameBuffer) {
		this->context->frameBuffer = &frameBuffer;
		context->numTilesX = Inline::ceilDiv(frameBuffer.getColorAttachment(0)->getWidth(), context->tileWidth);
		context->numTilesY = Inline::ceilDiv(frameBuffer.getColorAttachment(0)->getHeight(), context->tileWidth);
		updateVectorCapacity();
	}

	IFRIT_APIDECL void TileRasterRenderer::bindVertexBuffer(const VertexBuffer& vertexBuffer) {
		this->context->vertexBuffer = &vertexBuffer;
		varyingBufferDirtyFlag = true;
	}

	IFRIT_APIDECL void TileRasterRenderer::bindFragmentShader(FragmentShader& fragmentShader) {
		this->context->fragmentShader = &fragmentShader;
		if (!fragmentShader.isThreadSafe) {
			for (int i = 0; i < context->numThreads; i++) {
				context->threadSafeFSOwningSection[i] = fragmentShader.getThreadLocalCopy();
				context->threadSafeFS[i] = context->threadSafeFSOwningSection[i].get();
			}
		}
		else {
			for (int i = 0; i < context->numThreads; i++) {
				context->threadSafeFS[i] = &fragmentShader;
			}
		}
	}

	IFRIT_APIDECL void TileRasterRenderer::bindUniformBuffer(int binding, int set, BufferManager::IfritBuffer pBuffer) {
		auto p = pBuffer.manager.lock();
		void* data;
		p->mapBufferMemory(pBuffer, &data);
		this->context->uniformMapping[{binding, set}] = data;
	}

	IFRIT_APIDECL void TileRasterRenderer::bindIndexBuffer(BufferManager::IfritBuffer indexBuffer) {
		auto p = indexBuffer.manager.lock();
		p->mapBufferMemory(indexBuffer, (void**)&this->context->indexBuffer);
	}

	IFRIT_APIDECL void TileRasterRenderer::bindVertexShader(VertexShader& vertexShader){
		this->context->owningVaryingDesc = std::make_unique<VaryingDescriptor>(std::move(vertexShader.getVaryingDescriptor()));
		bindVertexShaderLegacy(vertexShader, *this->context->owningVaryingDesc);
	}

	IFRIT_APIDECL void TileRasterRenderer::bindVertexShaderLegacy(VertexShader& vertexShader, VaryingDescriptor& varyingDescriptor) {
		this->context->vertexShader = &vertexShader;
		this->context->varyingDescriptor = &varyingDescriptor;
		shaderBindingDirtyFlag = true;
		varyingBufferDirtyFlag = true;
		if (!vertexShader.isThreadSafe) {
			for (int i = 0; i < context->numThreads; i++) {
				context->threadSafeVSOwningSection[i] = vertexShader.getThreadLocalCopy();
				context->threadSafeVS[i] = context->threadSafeVSOwningSection[i].get();
			}
		}
		else {
			for (int i = 0; i < context->numThreads; i++) {
				context->threadSafeVS[i] = &vertexShader;
			}
		}
	}

	IFRIT_APIDECL void TileRasterRenderer::intializeRenderContext() {
		if (varyingBufferDirtyFlag) {
			context->vertexShaderResult = std::make_unique<VertexShaderResult>(
			context->vertexBuffer->getVertexCount(), context->varyingDescriptor->getVaryingCounts());
			shaderBindingDirtyFlag = false;
		}
		if (varyingBufferDirtyFlag) {
			context->vertexShaderResult->allocateVaryings(context->varyingDescriptor->getVaryingCounts());
			context->varyingDescriptor->applyVaryingDescriptors(context->vertexShaderResult.get());
			context->vertexShaderResult->setVertexCount(context->vertexBuffer->getVertexCount());
			varyingBufferDirtyFlag = false;
		}
	}

	void TileRasterRenderer::createWorkers() {
		workers.resize(context->numThreads);
		context->workerIdleTime.resize(context->numThreads);	
		for (int i = 0; i < context->numThreads; i++) {
			workers[i] = std::make_unique<TileRasterWorker>(i, shared_from_this(), context);
			workers[i]->status.store(TileRasterStage::CREATED, std::memory_order::relaxed);
			context->workerIdleTime[i] = 0;
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
	void TileRasterRenderer::statusTransitionBarrier2(TileRasterStage waitOn, TileRasterStage proceedTo) {
		while (true) {
			bool allOnBarrier = true;
			for (auto& worker : workers) {
				auto expected = waitOn;
				allOnBarrier = allOnBarrier && 
					(worker->status.compare_exchange_weak(expected, proceedTo, std::memory_order::acq_rel) ||
					(expected >= proceedTo));
			}
			if (allOnBarrier) break;
			std::this_thread::yield();
		}
	}
	
	IFRIT_APIDECL void TileRasterRenderer::setDepthFunc(IfritCompareOp depthFunc) {
		context->depthFuncSaved = depthFunc;
		if (context->optDepthTestEnableII) {
			context->depthFunc = depthFunc;
		}
	}
	IFRIT_APIDECL void TileRasterRenderer::setBlendFunc(IfritColorAttachmentBlendState state) {
		context->blendState = state;
		const auto& bs = state;
		if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
			context->blendColorCoefs.s = { 1,0,0,0 };
		}
		else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
			context->blendColorCoefs.s = { 0,0,0,1 };
		}
		else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
			context->blendColorCoefs.s = { 0,0,1,0 };
		}
		else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
			context->blendColorCoefs.s = { 0,1,0,0 };
		}
		else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
			context->blendColorCoefs.s = { 1,0,-1,0 };
		}
		else if (bs.srcColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
			context->blendColorCoefs.s = { 1,-1,0,0 };
		}
		else {
			ifritError("Unsupported blend factor");
		}
		//SrcAlpha
		if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
			context->blendAlphaCoefs.s = { 1,0,0,0 };
		}
		else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
			context->blendAlphaCoefs.s = { 0,0,0,1 };
		}
		else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
			context->blendAlphaCoefs.s = { 0,0,1,0 };
		}
		else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
			context->blendAlphaCoefs.s = { 0,1,0,0 };
		}
		else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
			context->blendAlphaCoefs.s = { 1,0,-1,0 };
		}
		else if (bs.srcAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
			context->blendAlphaCoefs.s = { 1,-1,0,0 };
		}
		else {
			ifritError("Unsupported blend factor");
		}

		//DstColor
		if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
			context->blendColorCoefs.d = { 1,0,0,0 };
		}
		else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
			context->blendColorCoefs.d = { 0,0,0,1 };
		}
		else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
			context->blendColorCoefs.d = { 0,0,1,0 };
		}
		else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
			context->blendColorCoefs.d = { 0,1,0,0 };
		}
		else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
			context->blendColorCoefs.d = { 1,0,-1,0 };
		}
		else if (bs.dstColorBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
			context->blendColorCoefs.d = { 1,-1,0,0 };
		}
		else {
			ifritError("Unsupported blend factor");
		}

		//DstAlpha
		if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE) {
			context->blendAlphaCoefs.d = { 1,0,0,0 };
		}
		else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ZERO) {
			context->blendAlphaCoefs.d = { 0,0,0,1 };
		}
		else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_DST_ALPHA) {
			context->blendAlphaCoefs.d = { 0,0,1,0 };
		}
		else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_SRC_ALPHA) {
			context->blendAlphaCoefs.d = { 0,1,0,0 };
		}
		else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) {
			context->blendAlphaCoefs.d = { 1,0,-1,0 };
		}
		else if (bs.dstAlphaBlendFactor == IfritBlendFactor::IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA) {
			context->blendAlphaCoefs.d = { 1,-1,0,0 };
		}
		else {
			ifritError("Unsupported blend factor");
		}
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
		auto totalTiles = context->numTilesX * context->numTilesY;
		if (counter >= totalTiles) {
			return -1;
		}
		return counter;
	}
	int TileRasterRenderer::fetchUnresolvedTileFragmentShading() {
		auto counter = unresolvedTileFragmentShading.fetch_add(1);
		auto totalTiles = context->numTilesX * context->numTilesY;
		if (counter >= totalTiles) {
			return -1;
		}
		return counter;
	}
	int TileRasterRenderer::fetchUnresolvedTileSort() {
		auto counter = unresolvedTileSort.fetch_add(1);
		auto totalTiles = context->numTilesX * context->numTilesY;
		if (counter >= totalTiles) {
			return -1;
		}
		return counter;
	}
	IFRIT_APIDECL void TileRasterRenderer::optsetForceDeterministic(bool opt) {
		context->optForceDeterministic = opt;
	}
	IFRIT_APIDECL void TileRasterRenderer::optsetDepthTestEnable(bool opt) {
		context->optDepthTestEnableII = opt;
		if (!opt) {
			context->depthFunc = IF_COMPARE_OP_ALWAYS;
		}
		else {
			context->depthFunc = context->depthFuncSaved;
		}
	}
	void TileRasterRenderer::resetWorkers(TileRasterStage expectedStage) {
		for (auto& worker : workers) {
			worker->status.store(expectedStage, std::memory_order::relaxed);
			worker->activated.store(true);
		}
	}
	void TileRasterRenderer::updateVectorCapacity() {
		auto totalTiles = context->numTilesX * context->numTilesY;
		context->sortedCoverQueue.resize(totalTiles);
		for (int i = 0; i < context->numThreads; i++) {
			context->rasterizerQueue[i].resize(totalTiles);
			context->coverQueue[i].resize(totalTiles);
		}
	}

	void TileRasterRenderer::updateUniformBuffer(){
		auto vsUniforms = context->vertexShader->getUniformList();
		auto fsUniforms = context->fragmentShader->getUniformList();
		for (int i = 0; i < context->numThreads; i++) {
			for (const auto& x : vsUniforms) {
				if (context->uniformMapping.count(x)) {
					context->threadSafeVS[i]->updateUniformData(x.first, x.second, context->uniformMapping[x]);
				}
			}
			for (const auto& x : fsUniforms) {
				if (context->uniformMapping.count(x)) {
					context->threadSafeFS[i]->updateUniformData(x.first, x.second, context->uniformMapping[x]);
				}
			}
		}
	}

	IFRIT_APIDECL void TileRasterRenderer::init() {
		context = std::make_shared<TileRasterContext>();
		context->rasterizerQueue.resize(context->numThreads);
		context->coverQueue.resize(context->numThreads);
		context->workerIdleTime.resize(context->numThreads);
		context->assembledTriangles.resize(context->numThreads);
		context->threadSafeFS.resize(context->numThreads);
		context->threadSafeVS.resize(context->numThreads);
		context->threadSafeFSOwningSection.resize(context->numThreads);
		context->threadSafeVSOwningSection.resize(context->numThreads);

		context->blendState.blendEnable = false;
		
		createWorkers();
		for (auto& worker : workers) {
			worker->status.store(TileRasterStage::CREATED, std::memory_order::relaxed);
			worker->threadStart();
		}
		initialized = true;
	}
	IFRIT_APIDECL void TileRasterRenderer::clear() {
		context->frameBuffer->getColorAttachment(0)->clearImageZero();
		context->frameBuffer->getDepthAttachment()->clearImage(255);
	}

	IFRIT_APIDECL void TileRasterRenderer::drawElements(int vertexCount, bool clearFramebuffer) IFRIT_AP_NOTHROW {
		intializeRenderContext();
		updateUniformBuffer();
		context->indexBufferSize = vertexCount;
		unresolvedTileRaster.store(0,std::memory_order::relaxed);
		unresolvedTileFragmentShading.store(0, std::memory_order::relaxed);
		unresolvedTileSort.store(0, std::memory_order::relaxed);
		if (clearFramebuffer) {
			resetWorkers(TileRasterStage::DRAWCALL_START_CLEAR);
		}
		else {
			resetWorkers(TileRasterStage::DRAWCALL_START);
		}
		statusTransitionBarrier(TileRasterStage::FRAGMENT_SHADING_SYNC, TileRasterStage::COMPLETED);
	}

}