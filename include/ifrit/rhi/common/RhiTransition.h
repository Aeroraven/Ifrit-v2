
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

#include "RhiBaseTypes.h"
#include "RhiResource.h"
#include "ifrit/core/base/containers/Atomic.h"

namespace Ifrit::RHI
{
    enum class ERhiTransitionState
    {
        Pending,
        Begin,
        End
    };

    struct RhiResourceTransitionDesc
    {
        ERhiResourceType    mType        = ERhiResourceType::Texture;
        RhiTexture*         mTexture     = nullptr;
        RhiBuffer*          mBuffer      = nullptr;
        RhiImageSubResource mSubResource = { 0, 0, 1, 1 };
        ERhiResourceState   mSrcState    = ERhiResourceState::Undefined;
        ERhiResourceState   mDstState    = ERhiResourceState::Undefined;
    };

    struct RhiTransition
    {
        TAtomic<ERhiTransitionState>   mState       = ERhiTransitionState::Pending;
        ERhiPipelineType               mPipelineSrc = ERhiPipelineType::Graphics;
        ERhiPipelineType               mPipelineDst = ERhiPipelineType::Graphics;
        Vec<RhiResourceTransitionDesc> mTransitions;

        Ref<RhiTaskSubmission>         mTransitionBeginSemaphore = nullptr;
    };

} // namespace Ifrit::RHI