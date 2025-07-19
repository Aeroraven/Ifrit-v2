#pragma once
#include "imgui.h"
#include <cmath>

namespace Ifrit::Editor::Internal
{
    inline ImVec4 LinearToSRGB(const ImVec4& color)
    {
        return ImVec4(powf(color.x, 2.2f), powf(color.y, 2.2f), powf(color.z, 2.2f), color.w);
    }

    inline void ApplyImGuiSytle()
    {
        // Code from https://github.com/ocornut/imgui/issues/707
        ImGuiStyle& style  = ImGui::GetStyle();
        ImVec4*     colors = style.Colors;

        // Primary background
        colors[ImGuiCol_WindowBg]  = LinearToSRGB(ImVec4(0.07f, 0.07f, 0.09f, 1.00f)); // #131318
        colors[ImGuiCol_MenuBarBg] = LinearToSRGB(ImVec4(0.12f, 0.12f, 0.15f, 1.00f)); // #131318

        colors[ImGuiCol_PopupBg] = LinearToSRGB(ImVec4(0.18f, 0.18f, 0.22f, 1.00f));

        // Headers
        colors[ImGuiCol_Header]        = LinearToSRGB(ImVec4(0.18f, 0.18f, 0.22f, 1.00f));
        colors[ImGuiCol_HeaderHovered] = LinearToSRGB(ImVec4(0.30f, 0.30f, 0.40f, 1.00f));
        colors[ImGuiCol_HeaderActive]  = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.35f, 1.00f));

        // Buttons
        colors[ImGuiCol_Button]        = LinearToSRGB(ImVec4(0.20f, 0.22f, 0.27f, 1.00f));
        colors[ImGuiCol_ButtonHovered] = LinearToSRGB(ImVec4(0.30f, 0.32f, 0.40f, 1.00f));
        colors[ImGuiCol_ButtonActive]  = LinearToSRGB(ImVec4(0.35f, 0.38f, 0.50f, 1.00f));

        // Frame BG
        colors[ImGuiCol_FrameBg]        = LinearToSRGB(ImVec4(0.15f, 0.15f, 0.18f, 1.00f));
        colors[ImGuiCol_FrameBgHovered] = LinearToSRGB(ImVec4(0.22f, 0.22f, 0.27f, 1.00f));
        colors[ImGuiCol_FrameBgActive]  = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.30f, 1.00f));

        // Tabs
        colors[ImGuiCol_Tab]                = LinearToSRGB(ImVec4(0.18f, 0.18f, 0.22f, 1.00f));
        colors[ImGuiCol_TabHovered]         = LinearToSRGB(ImVec4(0.35f, 0.35f, 0.50f, 1.00f));
        colors[ImGuiCol_TabActive]          = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.38f, 1.00f));
        colors[ImGuiCol_TabUnfocused]       = LinearToSRGB(ImVec4(0.13f, 0.13f, 0.17f, 1.00f));
        colors[ImGuiCol_TabUnfocusedActive] = LinearToSRGB(ImVec4(0.20f, 0.20f, 0.25f, 1.00f));

        // Title
        colors[ImGuiCol_TitleBg]          = LinearToSRGB(ImVec4(0.12f, 0.12f, 0.15f, 1.00f));
        colors[ImGuiCol_TitleBgActive]    = LinearToSRGB(ImVec4(0.15f, 0.15f, 0.20f, 1.00f));
        colors[ImGuiCol_TitleBgCollapsed] = LinearToSRGB(ImVec4(0.10f, 0.10f, 0.12f, 1.00f));

        // Borders
        colors[ImGuiCol_Border]       = LinearToSRGB(ImVec4(0.20f, 0.20f, 0.25f, 0.50f));
        colors[ImGuiCol_BorderShadow] = LinearToSRGB(ImVec4(0.00f, 0.00f, 0.00f, 0.00f));

        // Text
        colors[ImGuiCol_Text]         = LinearToSRGB(ImVec4(0.90f, 0.90f, 0.95f, 1.00f));
        colors[ImGuiCol_TextDisabled] = LinearToSRGB(ImVec4(0.50f, 0.50f, 0.55f, 1.00f));

        // Highlights
        colors[ImGuiCol_CheckMark]         = LinearToSRGB(ImVec4(0.50f, 0.70f, 1.00f, 1.00f));
        colors[ImGuiCol_SliderGrab]        = LinearToSRGB(ImVec4(0.50f, 0.70f, 1.00f, 1.00f));
        colors[ImGuiCol_SliderGrabActive]  = LinearToSRGB(ImVec4(0.60f, 0.80f, 1.00f, 1.00f));
        colors[ImGuiCol_ResizeGrip]        = LinearToSRGB(ImVec4(0.50f, 0.70f, 1.00f, 0.50f));
        colors[ImGuiCol_ResizeGripHovered] = LinearToSRGB(ImVec4(0.60f, 0.80f, 1.00f, 0.75f));
        colors[ImGuiCol_ResizeGripActive]  = LinearToSRGB(ImVec4(0.70f, 0.90f, 1.00f, 1.00f));

        // Scrollbar
        colors[ImGuiCol_ScrollbarBg]          = LinearToSRGB(ImVec4(0.10f, 0.10f, 0.12f, 1.00f));
        colors[ImGuiCol_ScrollbarGrab]        = LinearToSRGB(ImVec4(0.30f, 0.30f, 0.35f, 1.00f));
        colors[ImGuiCol_ScrollbarGrabHovered] = LinearToSRGB(ImVec4(0.40f, 0.40f, 0.50f, 1.00f));
        colors[ImGuiCol_ScrollbarGrabActive]  = LinearToSRGB(ImVec4(0.45f, 0.45f, 0.55f, 1.00f));

        // Style tweaks
        style.WindowRounding    = 5.0f;
        style.FrameRounding     = 5.0f;
        style.GrabRounding      = 5.0f;
        style.TabRounding       = 5.0f;
        style.PopupRounding     = 5.0f;
        style.ScrollbarRounding = 5.0f;
        style.WindowPadding     = ImVec2(10, 10);
        style.FramePadding      = ImVec2(6, 4);
        style.ItemSpacing       = ImVec2(8, 6);
        style.PopupBorderSize   = 0.f;
    }
} // namespace Ifrit::Editor::Internal
