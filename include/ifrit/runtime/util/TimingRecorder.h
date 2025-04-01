
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
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/platform/ApiConv.h"

namespace Ifrit::Runtime
{

    class IFRIT_APIDECL TimingRecorder
    {
    private:
        u64 m_curSystemTimeUs;
        u64 m_curTimeUs;
        u64 m_deltaTimeUs;

    public:
        void       OnUpdate();

        inline u64 GetCurSystemTimeUs() const { return m_curSystemTimeUs; }
        inline u64 GetCurTimeUs() const { return m_curTimeUs; }
        inline u64 GetDeltaTimeUs() const { return m_deltaTimeUs; }
    };

} // namespace Ifrit::Runtime