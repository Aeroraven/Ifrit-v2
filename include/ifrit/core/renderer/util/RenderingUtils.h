
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
#include "ifrit/common/math/constfunc/ConstFunc.h"
#include "ifrit/common/util/FileOps.h"
#include "ifrit/core/renderer/RendererUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <algorithm>
#include <bit>

#include "ifrit.shader/Syaro/Syaro.SharedConst.h"

namespace Ifrit::Core::RenderingUtil
{

    // These functions are just used to reduce code duplication. Although
    // "RenderFeature" and "RenderPass" designs might be better, the current design
    // is intended to simplify the codebase.

    IFRIT_APIDECL Graphics::Rhi::RhiShader* LoadShaderFromFile(Graphics::Rhi::RhiBackend* rhi,
        const char* shaderPath, const char* entryPoint,
        Graphics::Rhi::RhiShaderStage stage);

    IFRIT_APIDECL Graphics::Rhi::RhiComputePass* CreateComputePass(Graphics::Rhi::RhiBackend* rhi,
        const char* shaderPath, u32 numBindlessDescs,
        u32 numPushConsts);

    IFRIT_APIDECL                                Graphics::Rhi::RhiGraphicsPass*
                                                 CreateGraphicsPass(Graphics::Rhi::RhiBackend* rhi, const char* vsPath, const char* fsPath, u32 numBindlessDescs,
                                                     u32 numPushConsts, const Graphics::Rhi::RhiRenderTargetsFormat& vFmts);

    IFRIT_APIDECL void                           EnqueueFullScreenPass(const Graphics::Rhi::RhiCommandList* cmd,
                                  Graphics::Rhi::RhiBackend*                                                rhi,
                                  Graphics::Rhi::RhiGraphicsPass*                                           pass,
                                  Graphics::Rhi::RhiRenderTargets*                                          rt,
                                  const Vec<Graphics::Rhi::RhiBindlessDescriptorRef*>&                      vBindlessDescs,
                                  const void* pPushConst, u32 numPushConsts);

    IFRIT_APIDECL void                           warpRenderTargets(Graphics::Rhi::RhiBackend* rhi, Graphics::Rhi::RhiTexture* vTex,
                                  Ref<Graphics::Rhi::RhiColorAttachment>& vCA,
                                  Ref<Graphics::Rhi::RhiRenderTargets>&   vRT);

} // namespace Ifrit::Core::RenderingUtil