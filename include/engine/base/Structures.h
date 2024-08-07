#pragma once
#include "engine/base/Constants.h"
namespace Ifrit::Engine {
	struct IfritExtent3D {
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t depth = 0;
	};
	struct IfritOffset3D {
		int32_t x;
		int32_t y;
		int32_t z;
	};
	struct IfritSamplerT {
		IfritFilter filterMode = IF_FILTER_NEAREST;
		IfritSamplerAddressMode addressModeU = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		IfritSamplerAddressMode addressModeV = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		IfritSamplerAddressMode addressModeW = IF_SAMPLER_ADDRESS_MODE_REPEAT;
		IfritBorderColor borderColor = IF_BORDER_COLOR_BLACK;
		bool anisotropyEnable = false;
		float maxAnisotropy = 1.0f;
	};
	struct IfritImageCreateInfo {
		IfritExtent3D extent;
		IfritImageTiling tilingMode = IF_IMAGE_TILING_LINEAR;
		uint32_t mipLevels = 0;
		uint32_t arrayLayers = 1;
	};
	struct IfritImageSubresourceLayers {
		uint32_t mipLevel = 0;
		uint32_t baseArrayLayer = 0;
	};
	struct IfritImageBlit {
		IfritImageSubresourceLayers srcSubresource;
		IfritExtent3D srcExtentSt;
		IfritExtent3D srcExtentEd;
		IfritImageSubresourceLayers dstSubresource;
		IfritExtent3D dstExtentSt;
		IfritExtent3D dstExtentEd;
	};
	struct IfritBufferImageCopy {
		uint32_t bufferOffset;
		IfritImageSubresourceLayers imageSubresource;
		IfritOffset3D imageOffset;
		IfritExtent3D imageExtent;
	};
	struct IfritColorAttachmentBlendState {
		bool blendEnable;
		IfritBlendFactor srcColorBlendFactor;
		IfritBlendFactor dstColorBlendFactor;
		IfritBlendFactor srcAlphaBlendFactor;
		IfritBlendFactor dstAlphaBlendFactor;
	};
}