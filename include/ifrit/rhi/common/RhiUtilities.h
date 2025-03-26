
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

namespace Ifrit::Graphics::Rhi
{
    class IFRIT_APIDECL RhiDeviceTimer
    {
    public:
        virtual void Start(const RhiCommandList* cmd) = 0;
        virtual void Stop(const RhiCommandList* cmd)  = 0;
        virtual f32  GetElapsedMs()                   = 0;
    };
} // namespace Ifrit::Graphics::Rhi