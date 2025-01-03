
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
#include "ifrit/softgraphics/core/definition/CoreExports.h"
namespace Ifrit::GraphicsBackend::SoftGraphics {
enum IfritPolygonMode {
  IF_POLYGON_MODE_FILL = 0,
  IF_POLYGON_MODE_LINE = 1,
  IF_POLYGON_MODE_POINT = 2
};
enum IfritCullMode {
  IF_CULL_MODE_NONE = 0,
  IF_CULL_MODE_FRONT = 1,
  IF_CULL_MODE_BACK = 2,
  IF_CULL_MODE_FRONT_AND_BACK = 3
};
enum IfritSamplerAddressMode {
  IF_SAMPLER_ADDRESS_MODE_REPEAT = 0,
  IF_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT = 1,
  IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2,
  IF_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3,
  IF_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE = 4,
};
enum IfritFilter { IF_FILTER_NEAREST = 0, IF_FILTER_LINEAR = 1 };
enum IfritBorderColor { IF_BORDER_COLOR_BLACK = 0, IF_BORDER_COLOR_WHITE = 1 };
enum IfritImageTiling {
  IF_IMAGE_TILING_LINEAR = 1,
  IF_IMAGE_TILING_OPTIMAL = 2
};
enum IfritFormat {
  IF_FORMAT_R8G8B8A8_UINT = 0,
  IF_FORMAT_R32G32B32A32_SFLOAT = 1
};
enum IfritBlendFactor {
  IF_BLEND_FACTOR_ZERO = 0,
  IF_BLEND_FACTOR_ONE = 1,
  IF_BLEND_FACTOR_SRC_COLOR = 2,
  IF_BLEND_FACTOR_ONE_MINUS_SRC_COLOR = 3,
  IF_BLEND_FACTOR_DST_COLOR = 4,
  IF_BLEND_FACTOR_ONE_MINUS_DST_COLOR = 5,
  IF_BLEND_FACTOR_SRC_ALPHA = 6,
  IF_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA = 7,
  IF_BLEND_FACTOR_DST_ALPHA = 8,
  IF_BLEND_FACTOR_ONE_MINUS_DST_ALPHA = 9
};
enum IfritCompareOp {
  IF_COMPARE_OP_NEVER = 0,
  IF_COMPARE_OP_LESS = 1,
  IF_COMPARE_OP_EQUAL = 2,
  IF_COMPARE_OP_LESS_OR_EQUAL = 3,
  IF_COMPARE_OP_GREATER = 4,
  IF_COMPARE_OP_NOT_EQUAL = 5,
  IF_COMPARE_OP_GREATER_OR_EQUAL = 6,
  IF_COMPARE_OP_ALWAYS = 7
};

enum IfritSampleCountFlagBits {
  IF_SAMPLE_COUNT_1_BIT = 1,
  IF_SAMPLE_COUNT_2_BIT = 2,
  IF_SAMPLE_COUNT_4_BIT = 4,
  IF_SAMPLE_COUNT_8_BIT = 8,
  IF_SAMPLE_COUNT_16_BIT = 16,
};
} // namespace Ifrit::GraphicsBackend::SoftGraphics