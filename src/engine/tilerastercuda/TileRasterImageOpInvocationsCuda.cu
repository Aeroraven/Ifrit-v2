#include "engine/tilerastercuda/TileRasterImageOpInvocationsCuda.cuh"
#include "engine/tilerastercuda/TileRasterCommonResourceCuda.cuh"
#include "engine/math/ShaderOpsCuda.cuh"

namespace Ifrit::Engine::TileRaster::CUDA::Invocation::Impl {
	struct BlitImageKernelArgs {
		int srcOff, srcId;
		uint32_t srcWid;
		uint32_t srcHei;
		int dstOff, dstId;
		uint32_t dstWid;
		uint32_t dstHei;
		uint32_t srcCx, srcCy, srcDx, srcDy;
		uint32_t dstCx, dstCy, dstDx, dstDy;
	};
	IFRIT_KERNEL void blitImageBilinearKernel(BlitImageKernelArgs arg) {
		float4* isrc = reinterpret_cast<float4*>(csTextures[arg.srcId]) + arg.srcOff;
		float4* idst = reinterpret_cast<float4*>(csTextures[arg.dstId]) + arg.dstOff;
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
		idst[(curDstX + arg.dstCx) + (curDstY + arg.dstCy) * arg.dstWid] = result;
		
	}
	IFRIT_KERNEL void blitImageNearestKernel(BlitImageKernelArgs arg) {
		float4* isrc = reinterpret_cast<float4*>(csTextures[arg.srcId]) + arg.srcOff;
		float4* idst = reinterpret_cast<float4*>(csTextures[arg.dstId]) + arg.dstOff;
		int curDstX = threadIdx.x + blockDim.x * blockIdx.x;
		int curDstY = threadIdx.y + blockDim.y * blockIdx.y;
		if (curDstY + arg.dstCy >= arg.dstDy || curDstX + arg.dstCx >= arg.dstDx) {
			return;
		}
		float percentX =  1.0f * curDstX / (arg.dstDx - arg.dstCx);
		float percentY = 1.0f * curDstY / (arg.dstDy - arg.dstCy);
		float srcCorX = percentX * (arg.srcDx - arg.srcCx) + arg.srcCx;
		float srcCorY = percentY * (arg.srcDy - arg.srcCy) + arg.srcCy;
		int sX = min((int)round(srcCorX), arg.srcWid - 1);
		int sY = min((int)round(srcCorY), arg.srcHei - 1);
		float4 result = isrc[sY * arg.srcWid + sX];
		idst[(curDstX + arg.dstCx) + (curDstY + arg.dstCy) * arg.dstWid] = result;
		//printf("%d %d -> %f %f %f %f\n", (curDstX + arg.dstCx), (curDstY + arg.dstCy), result.x, result.y, result.z, result.w);

	}

	static int64_t sTestSortVal[1000000];
	static int64_t sTestSortVal2[1000000];

	constexpr int totalGroups = 128;
	IFRIT_DEVICE static int64_t hTestSortVal[1000000];
	static int sRegionStart[totalGroups];
	static int sRegionSize[totalGroups];
	IFRIT_DEVICE static int hRegionStart[totalGroups];
	IFRIT_DEVICE static int hRegionSize[totalGroups];

	template<class T>
	IFRIT_DEVICE void doInsertionSort(T* keys, int count) {
		for (int i = 1; i < count; i++) {
			T key = keys[i];
			int j = i - 1;
			while (j >= 0 && keys[j] > key) {
				keys[j + 1] = keys[j];
				j--;
			}
			keys[j + 1] = key;
		}
	}

	IFRIT_KERNEL void testSortImpl() {
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		int regionStart = hRegionStart[tid];
		int regionSize = hRegionSize[tid];
		doInsertionSort(hTestSortVal + regionStart, regionSize);
		//printf("%lld %d %d\n", (hTestSortVal + regionStart)[0], regionStart, regionSize);
	}


}

namespace Ifrit::Engine::TileRaster::CUDA::Invocation {
	void invokeBlitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter) {
		auto getMipLvlOffset = [&](int slotId, int mipLevel) {
			int baseOff = 0;
			int bw = Impl::hsTextureWidth[slotId];
			int bh = Impl::hsTextureHeight[slotId];
			for (int i = 0; i < mipLevel; i++) {
				baseOff += bw * bh;
				bw = (bw + 1) >> 1;
				bh = (bh + 1) >> 1;
			}
			return baseOff;
		};
		Impl::BlitImageKernelArgs blitArgs;
		blitArgs.dstCx = region.dstExtentSt.width;
		blitArgs.dstCy = region.dstExtentSt.height;
		blitArgs.dstDx = region.dstExtentEd.width;
		blitArgs.dstDy = region.dstExtentEd.height;
		blitArgs.srcCx = region.srcExtentSt.width;
		blitArgs.srcCy = region.srcExtentSt.height;
		blitArgs.srcDx = region.srcExtentEd.width;
		blitArgs.srcDy = region.srcExtentEd.height;
		blitArgs.dstHei = IFRIT_InvoCeilRshift(Impl::hsTextureHeight[dstSlotId], region.dstSubresource.mipLevel);
		blitArgs.dstWid = IFRIT_InvoCeilRshift(Impl::hsTextureWidth[dstSlotId], region.dstSubresource.mipLevel);
		blitArgs.dstOff = getMipLvlOffset(dstSlotId, region.dstSubresource.mipLevel);
		blitArgs.dstId = dstSlotId;
		blitArgs.srcHei = IFRIT_InvoCeilRshift(Impl::hsTextureHeight[srcSlotId], region.srcSubresource.mipLevel);
		blitArgs.srcWid = IFRIT_InvoCeilRshift(Impl::hsTextureWidth[srcSlotId], region.srcSubresource.mipLevel);
		blitArgs.srcOff = getMipLvlOffset(srcSlotId, region.srcSubresource.mipLevel);
		blitArgs.srcId = srcSlotId;
		int dw = blitArgs.dstDx - blitArgs.dstCx;
		int dh = blitArgs.dstDy - blitArgs.dstCy;
		int blockX = IFRIT_InvoGetThreadBlocks(dw, 8);
		int blockY = IFRIT_InvoGetThreadBlocks(dh, 8);
		if (filter == IF_FILTER_LINEAR) {
			Impl::blitImageBilinearKernel CU_KARG2(dim3(blockX, blockY, 1), dim3(8, 8, 1)) (blitArgs);
		}
		else if (filter == IF_FILTER_NEAREST) {
			Impl::blitImageNearestKernel CU_KARG2(dim3(blockX, blockY, 1), dim3(8, 8, 1)) (blitArgs);
		}
		cudaDeviceSynchronize();
		
	}
	void invokeMipmapGeneration(int slotId, IfritFilter filter) {
		int totalMipLevels = Impl::hsTextureMipLevels[slotId];
		uint32_t wid = Impl::hsTextureWidth[slotId];
		uint32_t hei = Impl::hsTextureHeight[slotId];
		IfritImageBlit region;
		region.srcExtentSt = { 0,0,0 };
		region.dstExtentSt = { 0,0,0 };
		for (int i = 0; i < totalMipLevels; i++) {
			uint32_t nw = (wid + 1) >> 1;
			uint32_t nh = (hei + 1) >> 1;
			region.srcExtentEd = { wid,hei,0 };
			region.dstExtentEd = { nw,nh,0 };
			region.srcSubresource.mipLevel = i;
			region.dstSubresource.mipLevel = i + 1;
			invokeBlitImage(slotId, slotId, region, filter);
			wid = nw;
			hei = nh;
		}
		cudaDeviceSynchronize();
	}

	void testSort() {
		int lastStart = 0;
		for(int i=0;i< Impl::totalGroups;i++){
			int regionSize = rand() % 10 + 1;
			Impl::sRegionSize[i] = regionSize;
			Impl::sRegionStart[i] = lastStart;
			lastStart += regionSize;
			for (int j = 0; j < regionSize; j++) {
				Impl::sTestSortVal[Impl::sRegionStart[i] + j] = rand();
			}
		}
		cudaMemcpyToSymbol(Impl::hRegionSize, Impl::sRegionSize, sizeof(int) * Impl::totalGroups);
		cudaMemcpyToSymbol(Impl::hRegionStart, Impl::sRegionStart, sizeof(int) * Impl::totalGroups);
		cudaMemcpyToSymbol(Impl::hTestSortVal, Impl::sTestSortVal, sizeof(int64_t) * 1000000);
		//Check error
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(error));
		}

		Impl::testSortImpl CU_KARG2(dim3(Impl::totalGroups /32, 1, 1), dim3(32, 1, 1)) ();
		cudaDeviceSynchronize();
		cudaMemcpyFromSymbol(Impl::sTestSortVal2, Impl::hTestSortVal, sizeof(int64_t) * 1000000);

		for (int i = 0; i < Impl::totalGroups; i++) {
			bool ilv = true;
			for (int j = 0; j < Impl::sRegionSize[i] - 1; j++) {
				if (Impl::sTestSortVal2[Impl::sRegionStart[i] + j] > Impl::sTestSortVal2[Impl::sRegionStart[i] + j + 1]) {
					ilv = false;
				}
			}
			if (true) {
				for (int j = 0; j < Impl::sRegionSize[i]; j++) {
					printf("%lld ", Impl::sTestSortVal[Impl::sRegionStart[i] + j]);
				}
				printf("->");
				for (int j = 0; j < Impl::sRegionSize[i]; j++) {
					printf("%lld ", Impl::sTestSortVal2[Impl::sRegionStart[i] + j]);
				}
				printf("\n");
			}
		}
	}
}