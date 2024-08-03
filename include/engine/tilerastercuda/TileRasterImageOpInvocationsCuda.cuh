#pragma once
#include "engine/base/Structures.h"
#ifdef IFRIT_FEATURE_CUDA
namespace Ifrit::Engine::TileRaster::CUDA::Invocation {
	void invokeBlitImage(int srcSlotId, int dstSlotId, const IfritImageBlit& region, IfritFilter filter);
	void invokeMipmapGeneration(int slotId, IfritFilter filter);
}
#endif