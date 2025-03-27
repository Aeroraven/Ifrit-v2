
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
    enum class InputKeyCode
    {
        Space        = 32,
        Num0         = 48,
        Num1         = 49,
        Num2         = 50,
        Num3         = 51,
        Num4         = 52,
        Num5         = 53,
        Num6         = 54,
        Num7         = 55,
        Num8         = 56,
        Num9         = 57,
        A            = 65,
        B            = 66,
        C            = 67,
        D            = 68,
        E            = 69,
        F            = 70,
        G            = 71,
        H            = 72,
        I            = 73,
        J            = 74,
        K            = 75,
        L            = 76,
        M            = 77,
        N            = 78,
        O            = 79,
        P            = 80,
        Q            = 81,
        R            = 82,
        S            = 83,
        T            = 84,
        U            = 85,
        V            = 86,
        W            = 87,
        X            = 88,
        Y            = 89,
        Z            = 90,
        Escape       = 256,
        Enter        = 257,
        Tab          = 258,
        Backspace    = 259,
        Insert       = 260,
        Delete       = 261,
        Right        = 262,
        Left         = 263,
        Down         = 264,
        Up           = 265,
        PageUp       = 266,
        PageDown     = 267,
        Home         = 268,
        End          = 269,
        CapsLock     = 280,
        ScrollLock   = 281,
        NumLock      = 282,
        PrintScreen  = 283,
        Pause        = 284,
        F1           = 290,
        F2           = 291,
        F3           = 292,
        F4           = 293,
        F5           = 294,
        F6           = 295,
        F7           = 296,
        F8           = 297,
        F9           = 298,
        F10          = 299,
        F11          = 300,
        F12          = 301,
        F13          = 302,
        F14          = 303,
        F15          = 304,
        F16          = 305,
        F17          = 306,
        F18          = 307,
        F19          = 308,
        F20          = 309,
        F21          = 310,
        F22          = 311,
        F23          = 312,
        F24          = 313,
        F25          = 314,
        Kp0          = 320,
        Kp1          = 321,
        Kp2          = 322,
        Kp3          = 323,
        Kp4          = 324,
        Kp5          = 325,
        Kp6          = 326,
        Kp7          = 327,
        Kp8          = 328,
        Kp9          = 329,
        KpDecimal    = 330,
        KpDivide     = 331,
        KpMultiply   = 332,
        KpSubtract   = 333,
        KpAdd        = 334,
        KpEnter      = 335,
        KpEqual      = 336,
        LeftShift    = 340,
        LeftControl  = 341,
        LeftAlt      = 342,
        LeftSuper    = 343,
        RightShift   = 344,
        RightControl = 345,
        RightAlt     = 346,
        RightSuper   = 347,
        Menu         = 348
    };

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
        bool IsKeyPressed(InputKeyCode key) { return m_keyStatus[static_cast<int>(key)].stat == 1; }
        bool IsKeyReleased(InputKeyCode key) { return m_keyStatus[static_cast<int>(key)].stat == 0; }
        void OnFrameUpdate();
        void UpdateKeyStatus(u32 key, u8 status) { m_keyStatus[key].stat = status; }

    private:
        void Init();
    };

} // namespace Ifrit::Core
