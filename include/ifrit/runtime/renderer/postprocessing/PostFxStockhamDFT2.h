
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/algo/Hash.h"
#include "ifrit/runtime/renderer/PostprocessPass.h"
#include <unordered_map>

namespace Ifrit::Runtime::PostprocessPassCollection
{

    class IFRIT_APIDECL PostFxStockhamDFT2 : public PostprocessPass
    {
        using GPUBindId     = Graphics::Rhi::RhiDescHandleLegacy;
        using RenderTargets = Graphics::Rhi::RhiRenderTargets;
        using GPUTexture    = Graphics::Rhi::RhiTextureRef;
        using ComputePass   = Graphics::Rhi::RhiComputePass;

        // I know this is UGLY
        CustomHashMap<Pair<u32, u32>, GPUTexture, PairwiseHash<u32, u32>> m_tex1;
        ComputePass*                                                      m_testBlurPipeline = nullptr;

    public:
        PostFxStockhamDFT2(IApplication* app);
        void RenderPostFx(
            const GPUCmdBuffer* cmd, GPUBindId* srcSampId, GPUBindId* dstUAVImg, u32 width, u32 height, u32 downscale);

    private:
        void RunCommand(const GPUCmdBuffer* cmd, u32 wgX, u32 wgY, const void* pc);
    };

} // namespace Ifrit::Runtime::PostprocessPassCollection