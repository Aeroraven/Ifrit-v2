#pragma once

#include "core/definition/CoreExports.h"
#include "engine/tileraster/TileRasterRenderer.h"

namespace Ifrit::Engine::TileRaster {
	class TileRasterWorker {
	public:
		std::atomic<TileRasterStage> status;
		std::atomic<bool> activated;
	private:
		std::unique_ptr<std::thread> execWorker;
		uint32_t workerId;
		std::shared_ptr<TileRasterRenderer> renderer;
		std::shared_ptr<TileRasterContext> context;
		bool vert= false;
		bool geom = false;
		bool rast = false;
		bool frag = false;

		

	public:
		TileRasterWorker(uint32_t workerId, std::shared_ptr<TileRasterRenderer> renderer, std::shared_ptr<TileRasterContext> context);
		void run();

		bool triangleFrustumClip(float4 v1, float4 v2, float4 v3, rect2Df& bbox);
		bool triangleCulling(float4 v1, float4 v2, float4 v3);
		void executeBinner(const int primitiveId, float4 v1, float4 v2, float4 v3, rect2Df bbox);

		void vertexProcessing();
		void geometryProcessing();
		void rasterization();
		void fragmentProcessing();


		void threadStart();
	};
}