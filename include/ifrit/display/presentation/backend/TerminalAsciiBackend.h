
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
#include "ifrit/display/presentation/backend/AbstractTerminalBackend.h"

namespace Ifrit::Display::Backend
{
    class IFRIT_APIDECL TerminalAsciiBackend : public AbstractTerminalBackend
    {
    private:
        i32                             consoleWidth;
        i32                             consoleHeight;
        IF_CONSTEXPR static const char* ramp = R"($@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:," ^ `'. )";
        String                          resultBuffer;

    public:
        TerminalAsciiBackend(i32 cWid, i32 cHeight);
        virtual void UpdateTexture(const f32* image, i32 channels, i32 width, i32 height) override;
        virtual void Draw() override;
        virtual void SetViewport(i32, i32, i32, i32) override {}
    };
} // namespace Ifrit::Display::Backend