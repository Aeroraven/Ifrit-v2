#pragma once
#include "core/definition/CoreExports.h"
#include "engine/tileraster/TileRasterContext.h"
#include "engine/bufferman/BufferManager.h"

namespace Ifrit::Engine::TileRaster {
	using namespace Ifrit::Engine;

	enum class TileRasterStage {
		IDLE,
		DRAWCALL_START,
		DRAWCALL_START_CLEAR,
		DRAWCALL_SYNC,
		VERTEX_SHADING,
		VERTEX_SHADING_SYNC,
		GEOMETRY_PROCESSING,
		GEOMETRY_PROCESSING_SYNC,
		RASTERIZATION,
		RASTERIZATION_SYNC,
		SORTING,
		SORTING_SYNC,
		FRAGMENT_SHADING,
		FRAGMENT_SHADING_SYNC,
		COMPLETED,
		TERMINATING
	};
	class TileRasterWorker;

	class TileRasterRenderer : public Renderer, public std::enable_shared_from_this<TileRasterRenderer> {

	private:
		bool shaderBindingDirtyFlag = true;
		bool varyingBufferDirtyFlag = true;
		std::shared_ptr<TileRasterContext> context;
		std::vector<std::unique_ptr<TileRasterWorker>> workers;
		std::unique_ptr<TileRasterWorker> selfOwningWorker;
		std::mutex lock;
		std::atomic<uint32_t> unresolvedTileRaster = 0;
		std::atomic<uint32_t> unresolvedTileFragmentShading = 0;
		std::atomic<uint32_t> unresolvedTileSort = 0;
		std::atomic<uint32_t> unresolvedChunkVertex = 0;
		std::atomic<uint32_t> unresolvedChunkGeometry = 0;

		bool initialized = false;

	protected:
		void createWorkers();
		void resetWorkers(TileRasterStage expectedStage);
		void statusTransitionBarrier2(TileRasterStage waitOn, TileRasterStage proceedTo);
		void statusTransitionBarrier3(TileRasterStage waitOn, TileRasterStage proceedTo);

		int fetchUnresolvedChunkVertex();
		int fetchUnresolvedChunkGeometry();

		int fetchUnresolvedTileRaster();
		int fetchUnresolvedTileFragmentShading();
		int fetchUnresolvedTileSort();

		void updateVectorCapacity();
		void updateUniformBuffer();

	public:
		friend class TileRasterWorker;
		IFRIT_APIDECL TileRasterRenderer();
		IFRIT_APIDECL ~TileRasterRenderer();
		IFRIT_APIDECL void bindFrameBuffer(FrameBuffer& frameBuffer);
		IFRIT_APIDECL void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		IFRIT_APIDECL void bindIndexBuffer(BufferManager::IfritBuffer indexBuffer);
		IFRIT_APIDECL void bindVertexShader(VertexShader& vertexShader);
		IFRIT_APIDECL void bindVertexShaderLegacy(VertexShader& vertexShader, VaryingDescriptor& varyingDescriptor);
		IFRIT_APIDECL void bindFragmentShader(FragmentShader& fragmentShader);
		IFRIT_APIDECL void bindUniformBuffer(int binding, int set, BufferManager::IfritBuffer pBuffer);
		IFRIT_APIDECL void setBlendFunc(IfritColorAttachmentBlendState state);
		IFRIT_APIDECL void setDepthFunc(IfritCompareOp depthFunc);

		IFRIT_APIDECL void intializeRenderContext();
		IFRIT_APIDECL void optsetForceDeterministic(bool opt);
		IFRIT_APIDECL void optsetDepthTestEnable(bool opt);

		IFRIT_APIDECL void drawElements(int vertexCount, bool clearFramebuffer) IFRIT_AP_NOTHROW;
		IFRIT_APIDECL void clear();
		IFRIT_APIDECL void init();
	};
}