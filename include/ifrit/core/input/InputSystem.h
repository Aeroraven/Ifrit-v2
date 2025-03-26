
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
#include "ifrit/core/base/ApplicationInterface.h"
#include <array>

namespace Ifrit::Core
{

    class IFRIT_APIDECL InputSystem
    {
    private:
        struct KeyStatus
        {
            u8 stat = 0;
        };
        enum class KeyStatusEnum
        {
            Pressed  = 1,
            Released = 0
        };
        Array<KeyStatus, 349> m_keyStatus;
        IApplication*         m_app;

    public:
        InputSystem(IApplication* app)
            : m_app(app) { Init(); }
        virtual ~InputSystem() = default;
        bool IsKeyPressed(u32 key) { return m_keyStatus[key].stat == 1; }
        bool IsKeyReleased(u32 key) { return m_keyStatus[key].stat == 0; }
        void OnFrameUpdate();
        void UpdateKeyStatus(u32 key, u8 status) { m_keyStatus[key].stat = status; }

    private:
        void Init();
    };

} // namespace Ifrit::Core
