
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/common/base/IfritBase.h"
#include "ifrit/softgraphics/engine/base/Constants.h"
namespace Ifrit::GraphicsBackend::SoftGraphics {
// Buffer
struct IFRIT_APIDECL IfritBufferCreateInfo {
  size_t bufferSize;
};

// Runtime Deps
struct IFRIT_APIDECL IfritRayDesc {
  Vector3f Origin;
  Vector3f Direction;
  float TMin;
  float TMax;
};

// Other
struct IFRIT_APIDECL IfritExtent3D {
  u32 width = 0;
  u32 height = 0;
  u32 depth = 0;
};
struct IFRIT_APIDECL IfritOffset3D {
  int32_t x;
  int32_t y;
  int32_t z;
};
struct IFRIT_APIDECL IfritSamplerT {
  IfritFilter filterMode = IF_FILTER_NEAREST;
  IfritSamplerAddressMode addressModeU = IF_SAMPLER_ADDRESS_MODE_REPEAT;
  IfritSamplerAddressMode addressModeV = IF_SAMPLER_ADDRESS_MODE_REPEAT;
  IfritSamplerAddressMode addressModeW = IF_SAMPLER_ADDRESS_MODE_REPEAT;
  IfritBorderColor borderColor = IF_BORDER_COLOR_BLACK;
  bool anisotropyEnable = false;
  float maxAnisotropy = 1.0f;
};
struct IFRIT_APIDECL IfritImageCreateInfo {
  IfritExtent3D extent;
  IfritImageTiling tilingMode = IF_IMAGE_TILING_LINEAR;
  u32 mipLevels = 0;
  u32 arrayLayers = 1;
  IfritFormat format = IF_FORMAT_R32G32B32A32_SFLOAT;
};
struct IFRIT_APIDECL IfritImageSubresourceLayers {
  u32 mipLevel = 0;
  u32 baseArrayLayer = 0;
};
struct IFRIT_APIDECL IfritImageBlit {
  IfritImageSubresourceLayers srcSubresource;
  IfritExtent3D srcExtentSt;
  IfritExtent3D srcExtentEd;
  IfritImageSubresourceLayers dstSubresource;
  IfritExtent3D dstExtentSt;
  IfritExtent3D dstExtentEd;
};
struct IFRIT_APIDECL IfritBufferImageCopy {
  u32 bufferOffset;
  IfritImageSubresourceLayers imageSubresource;
  IfritOffset3D imageOffset;
  IfritExtent3D imageExtent;
};
struct IFRIT_APIDECL IfritColorAttachmentBlendState {
  bool blendEnable;
  IfritBlendFactor srcColorBlendFactor;
  IfritBlendFactor dstColorBlendFactor;
  IfritBlendFactor srcAlphaBlendFactor;
  IfritBlendFactor dstAlphaBlendFactor;
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics