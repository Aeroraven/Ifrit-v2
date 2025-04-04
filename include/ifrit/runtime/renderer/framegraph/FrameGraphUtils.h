
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

#pragma once
#include "ifrit/runtime/renderer/framegraph/FrameGraph.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime::FrameGraphUtils
{
    IFRIT_RUNTIME_API GraphicsPassNode& AddFullScreenQuadPass(FrameGraphBuilder& builder, const String& name,
        const String& vs, const String& fs, Graphics::Rhi::RhiRenderTargets* rts, const void* ptr, u32 pushConsts);

    IFRIT_RUNTIME_API ComputePassNode&  AddComputePass(FrameGraphBuilder& builder, const String& name,
         const String& shader, Vector3i workGroups, const void* ptr, u32 pushConsts);

} // namespace Ifrit::Runtime::FrameGraphUtils