#pragma once
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/runtime/application/Subsystem.h"
#include "ifrit/runtime/forwarding/FwdBase.h"
#include "ifrit/runtime/base/Base.h"

namespace Ifrit::Runtime
{
    enum class EInputMouseButton : u8
    {
        Left   = 1,
        Right  = 2,
        Middle = 3
    };
    enum class EInputKeyCode
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

    class IFRIT_RUNTIME_API InputSystem : public ISubsystem
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
        Array<KeyStatus, 3>   mMouseButtonStatus;
        IApplication*         m_app;
        float                 mMouseX = 0.0f;
        float                 mMouseY = 0.0f;

    public:
        virtual ~InputSystem();

        bool                                  IsKeyPressed(EInputKeyCode key);
        bool                                  IsKeyReleased(EInputKeyCode key);
        bool                                  IsMouseButtonPressed(EInputMouseButton button);
        bool                                  IsMouseButtonReleased(EInputMouseButton button);
        float                                 GetMouseX() const;
        float                                 GetMouseY() const;

        void                                  OnFrameUpdate();
        void                                  UpdateKeyStatus(u32 key, u8 status);
        void                                  UpdateMousePosition(float x, float y);
        void                                  UpdateMouseButtonStatus(u32 button, u8 status);

        virtual void                          OnInitialize(IApplication* app) override;
        virtual void                          OnShutdown() override;
        virtual void                          OnFrameBegin() override;
        virtual void                          OnFrameEnd() override;
        virtual Owner<RHI::RhiTaskSubmission> OnPreRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual Owner<RHI::RhiTaskSubmission> OnPostRendering(RHI::RhiTaskSubmission* prevSubmission) override;
        virtual void                          OnUpdate(Scene* scene) override;

        static Owner<InputSystem>             Create();
    };

} // namespace Ifrit::Runtime
