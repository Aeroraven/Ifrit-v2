#ifdef IFRIT_FEATURE_CUDA
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterImageOpInvocationsCuda.cuh"
#include "ifrit/softgraphics/engine/tilerastercuda/TileRasterCommonResourceCuda.cuh"
#include "ifrit/softgraphics/engine/math/ShaderOpsCuda.cuh"

namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation::Impl {
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
		
		using Ifrit::SoftRenderer::Math::ShaderOps::CUDA::lerp;
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
	}

}

namespace Ifrit::SoftRenderer::TileRaster::CUDA::Invocation {
	void invokeBlitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter) {
		auto getMipLvlOffset = [&](int slotId, int mipLevel,int totalLayers,int dstLayers) {
			int baseOff = 0;
			int bw = Impl::hsTextureWidth[slotId];
			int bh = Impl::hsTextureHeight[slotId];
			for (int i = 0; i < mipLevel; i++) {
				baseOff += bw * bh * totalLayers;
				bw = (bw + 1) >> 1;
				bh = (bh + 1) >> 1;
			}
			baseOff += bw * bh * dstLayers;
			return baseOff;
		};
		auto dstTotalLayers = Impl::hsTextureArrayLayers[dstSlotId];
		auto srcTotalLayers = Impl::hsTextureArrayLayers[srcSlotId];
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
		blitArgs.dstOff = getMipLvlOffset(dstSlotId, region.dstSubresource.mipLevel, dstTotalLayers, region.dstSubresource.baseArrayLayer);
		blitArgs.dstId = dstSlotId;
		blitArgs.srcHei = IFRIT_InvoCeilRshift(Impl::hsTextureHeight[srcSlotId], region.srcSubresource.mipLevel);
		blitArgs.srcWid = IFRIT_InvoCeilRshift(Impl::hsTextureWidth[srcSlotId], region.srcSubresource.mipLevel);
		blitArgs.srcOff = getMipLvlOffset(srcSlotId, region.srcSubresource.mipLevel, srcTotalLayers, region.dstSubresource.baseArrayLayer);
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
		int totalArrLayers = Impl::hsTextureArrayLayers[slotId];
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
			for (int j = 0; j < totalArrLayers; j++) {
				region.srcSubresource.baseArrayLayer = j;
				region.dstSubresource.baseArrayLayer = j;
				invokeBlitImage(slotId, slotId, region, filter);
			}
			wid = nw;
			hei = nh;
		}
		cudaDeviceSynchronize();
	}
	void invokeCopyBufferToImage(void* srcBuffer, int dstImage, uint32_t regionCount, const IfritBufferImageCopy* pRegions) {
		auto texH = Impl::hsTextureHeight[dstImage];
		auto texW = Impl::hsTextureWidth[dstImage];
		auto copyFunc = [&](const IfritBufferImageCopy& region) {
			if (region.imageOffset.x != 0 || region.imageOffset.y != 0 || region.imageOffset.z != 0) {
				printf("Copy with offset is not supported now\n");
				std::abort();
			}
			if (region.imageExtent.height != texH || region.imageExtent.width != texW || region.imageExtent.depth != 1) {
				printf("Partial image copy is not supported now\n");
				std::abort();
			}
			if (region.imageSubresource.mipLevel != 0) {
				printf("Copy to mip levels is not supported now\n");
				std::abort();
			}
			int offset = texH * texW * region.imageSubresource.baseArrayLayer * 4;
			int totalSize = texH * texW * sizeof(float4);
			cudaMemcpy(Impl::hsTextures[dstImage]+offset, (char*)srcBuffer+region.bufferOffset, totalSize, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
		};
		for (int i = 0; i < regionCount; i++) {
			copyFunc(pRegions[i]);
		}
	}
}
#endif