#pragma once
#include "engine/base/Structures.h"
#ifdef IFRIT_FEATURE_CUDA
namespace Ifrit::Engine::SoftRenderer::TileRaster::CUDA::Invocation {
	void invokeBlitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter);
	void invokeMipmapGeneration(int slotId, IfritFilter filter);
	void invokeCopyBufferToImage(void* srcBuffer, int dstImage, uint32_t regionCount, const IfritBufferImageCopy* pRegions);
}
#endif