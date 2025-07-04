
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
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/core/file/FileOps.h"
#include "ifrit/runtime/renderer/RendererUtil.h"
#include <algorithm>
#include <bit>

#include "ifrit/runtime/base/ApplicationInterface.h"
#include "ifrit/runtime/material/ShaderVariantDescriptor.h"

namespace Ifrit::Runtime::RenderingUtil
{

    // These functions are just used to reduce code duplication. Although
    // "RenderFeature" and "RenderPass" designs might be better, the current design
    // is intended to simplify the codebase.

    IFRIT_APIDECL RHI::RhiComputePass* CreateComputePassInternal(
        IApplication* app, const ShaderVariantDesc& desc, u32 numBindlessDescs, u32 numPushConsts);

    IFRIT_APIDECL RHI::RhiGraphicsPass* CreateGraphicsPassInternal(IApplication* app, const ShaderVariantDesc& vsDesc,
        const ShaderVariantDesc& fsDesc, u32 numBindlessDescs, u32 numPushConsts,
        const RHI::RhiRenderTargetsFormat& vFmts);

    IFRIT_APIDECL void                  EnqueueFullScreenPass(const RHI::RhiCommandList* cmd, RHI::RhiBackend* rhi,
                         RHI::RhiGraphicsPass* pass, RHI::RhiRenderTargets* rt,
                         const Vec<RHI::RhiBindlessDescriptorRef*>& vBindlessDescs, const void* pPushConst, u32 numPushConsts);

    IFRIT_APIDECL void WarpRenderTargets(RHI::RhiBackend* rhi, RHI::RhiTexture* vTex, Ref<RHI::RhiColorAttachment>& vCA,
        Ref<RHI::RhiRenderTargets>& vRT);

} // namespace Ifrit::Runtime::RenderingUtil