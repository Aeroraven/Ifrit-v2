
/*
Ifrit-v2
Copyright (C) 2024-2025 funkybirds(Aeroraven)

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

// Core functions are always called by many modules.
// So, this file uses pch to reduce compile time.

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"
#include "ifrit/core/typing/Util.h"
#include "ifrit/core/typing/CountRef.h"

#include "ifrit/core/algo/Hash.h"
#include "ifrit/core/algo/Identifier.h"
#include "ifrit/core/algo/Memory.h"
#include "ifrit/core/algo/Container.h"
#include "ifrit/core/algo/Parallel.h"

#include "ifrit/core/math/VectorDefs.h"
#include "ifrit/core/math/VectorOps.h"
#include "ifrit/core/math/LinalgOps.h"
#include "ifrit/core/math/GeometryFunctions.h"
#include "ifrit/core/math/SphericalSampling.h"
#include "ifrit/core/math/simd/SimdVectors.h"
#include "ifrit/core/math/fastutil/FastUtil.h"
#include "ifrit/core/math/fastutil/FastUtilSimd.h"
#include "ifrit/core/math/constfunc/ConstFunc.h"

#include "ifrit/core/file/FileOps.h"

#include "ifrit/core/logging/Logging.h"
