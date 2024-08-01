#pragma once
#include "core/definition/CoreExports.h"
#include "engine/tileraster/TileRasterContext.h"

namespace Ifrit::Engine::TileRaster {
	using namespace Ifrit::Engine;

	enum class TileRasterStage {
		CREATED,
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
		TERMINATED
	};

	class TileRasterWorker;

	class TileRasterRenderer : public Renderer, public std::enable_shared_from_this<TileRasterRenderer> {

	private:
		bool shaderBindingDirtyFlag = true;
		bool varyingBufferDirtyFlag = true;
		std::shared_ptr<TileRasterContext> context;
		std::vector<std::unique_ptr<TileRasterWorker>> workers;
		std::mutex lock;
		std::atomic<uint32_t> unresolvedTileRaster = 0;
		std::atomic<uint32_t> unresolvedTileFragmentShading = 0;
		std::atomic<uint32_t> unresolvedTileSort = 0;

	protected:
		void createWorkers();
		void resetWorkers();
		void statusTransitionBarrier(TileRasterStage waitOn, TileRasterStage proceedTo);
		void waitOnWorkers(TileRasterStage waitOn);
		
		int fetchUnresolvedTileRaster();
		int fetchUnresolvedTileFragmentShading();
		int fetchUnresolvedTileSort();

		void updateVectorCapacity();

	public:
		friend class TileRasterWorker;
		TileRasterRenderer();
		void bindFrameBuffer(FrameBuffer& frameBuffer);
		void bindVertexBuffer(const VertexBuffer& vertexBuffer);
		void bindIndexBuffer(const std::vector<int>& indexBuffer);
		void bindVertexShader(VertexShader& vertexShader, VaryingDescriptor& varyingDescriptor);
		void bindFragmentShader(FragmentShader& fragmentShader);
		void setBlendFunc(IfritColorAttachmentBlendState state);

		void intializeRenderContext();
		void optsetForceDeterministic(bool opt);
		void optsetDepthTestEnable(bool opt);

		void render(bool clearFramebuffer) IFRIT_AP_NOTHROW;
		void clear();
		void init();
	};
}