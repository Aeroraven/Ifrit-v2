
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
#include "ifrit/display/presentation/backend/AbstractTerminalBackend.h"

namespace Ifrit::Display::Backend
{
    class IFRIT_APIDECL TerminalCharColorBackend : public AbstractTerminalBackend
    {
    private:
        int    consoleWidth;
        int    consoleHeight;
        String resultBuffer;

    public:
        TerminalCharColorBackend(int cWid, int cHeight);
        virtual void UpdateTexture(const float* image, int channels, int width, int height) override;
        virtual void Draw() override;
        virtual void SetViewport(i32, i32, i32, i32) override {}
    };
} // namespace Ifrit::Display::Backend