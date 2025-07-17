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
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/renderer/framegraph/FrameGraphUtils.h"

namespace Ifrit::Runtime::Artemis
{
    struct APICHostTestPrivateData;

    class IFRIT_RUNTIME_API APICHostTest
    {
    private:
        APICHostTestPrivateData* m_Data = nullptr;

    private:
        u32            GridToIndex(u32 x, u32 y);
        Pair<u32, u32> IndexToGrid(u32 index);

    public:
        APICHostTest();
        ~APICHostTest();

        void TestAPIC(FrameGraphBuilder& builder, FGTextureNode* renderTarget);
    };
} // namespace Ifrit::Runtime::Artemis