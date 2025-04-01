
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

#include "ifrit/runtime/util/TimingRecorder.h"
#include <chrono>

namespace Ifrit::Runtime
{

    IFRIT_APIDECL void TimingRecorder::OnUpdate()
    {
        auto currentSysTimeUs =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
        u64 currentSysTimeUsI64 = currentSysTimeUs.count();
        if (m_curSystemTimeUs != 0)
        {
            auto deltaTimeUs = currentSysTimeUsI64 - m_curSystemTimeUs;
            m_deltaTimeUs    = deltaTimeUs;
            m_curTimeUs += deltaTimeUs;
        }
        m_curSystemTimeUs = currentSysTimeUsI64;
    }

} // namespace Ifrit::Runtime