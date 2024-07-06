#include "engine/tilerastercuda/TileRasterImageOpInvocationsCuda.cuh"
#include "engine/tilerastercuda/TileRasterCommonResourceCuda.cuh"
#include "engine/math/ShaderOpsCuda.cuh"

namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	struct BlitImageKernelArgs {
		float* srcPtr;
		uint32_t srcWid;
		uint32_t srcHei;
		float* dstPtr;
		uint32_t dstWid;
		uint32_t dstHei;
		uint32_t srcCx, srcCy, srcDx, srcDy;
		uint32_t dstCx, dstCy, dstDx, dstDy;
	};
	IFRIT_KERNEL void blitImageBilinearKernel(BlitImageKernelArgs arg) {
		float4* isrc = reinterpret_cast<float4*>(arg.srcPtr);
		float4* idst = reinterpret_cast<float4*>(arg.dstPtr);
		int curDstX = threadIdx.x + blockDim.x * blockIdx.x;
		int curDstY = threadIdx.y + blockDim.y * blockIdx.y;
		if (curDstY + arg.dstCy >= arg.dstDy || curDstX + arg.dstCx >= arg.dstDx) {
			return;
		}
		float percentX = 1.0f * curDstX / (arg.dstDx - arg.dstCx);
		float percentY = 1.0f * curDstY / (arg.dstDy - arg.dstCy);
		float srcCorX = percentX * (arg.srcDx - arg.srcCx) + arg.srcCx;
		float srcCorY = percentY * (arg.srcDy - arg.srcCy) + arg.srcCy;

		int srcIntX = (int)srcCorX;
		int srcIntY = (int)srcCorY;
		float pX = srcCorX - srcIntX;
		float pY = srcCorY - srcIntY;

		float4 c00 = isrc[srcIntY * arg.srcWid + srcIntX];
		float4 c01 = isrc[srcIntY * arg.srcWid + min(srcIntX + 1, arg.srcWid - 1)];
		float4 c10 = isrc[min(srcIntY + 1, arg.srcHei - 1) * arg.srcWid + srcIntX];
		float4 c11 = isrc[min(srcIntY + 1, arg.srcHei - 1) * arg.srcWid + min(srcIntX + 1, arg.srcWid - 1)];
		
		using Ifrit::Engine::Math::ShaderOps::CUDA::lerp;
		float4 c0x = lerp(c00, c01, pX);
		float4 c1x = lerp(c10, c11, pX);
		float4 result = lerp(c0x, c1x, pY);
		idst[(curDstX + arg.dstCx) * arg.dstWid + (curDstY + arg.dstCy)] = result;
	}
}

namespace Ifrit::Engine::TileRaster::CUDA::Invocation {
	void invokeBlitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter) {
		
	}
}