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

        /// 0 = FLAT APPEARENCE
        /// 1 = MORE "3D" LOOK
        int         is3D = 0;

        colors[ImGuiCol_Text]                  = LinearToSRGB(ImVec4(1.00f, 1.00f, 1.00f, 1.00f));
        colors[ImGuiCol_TextDisabled]          = LinearToSRGB(ImVec4(0.40f, 0.40f, 0.40f, 1.00f));
        colors[ImGuiCol_ChildBg]               = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.25f, 1.00f));
        colors[ImGuiCol_WindowBg]              = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.25f, 1.00f));
        colors[ImGuiCol_PopupBg]               = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.25f, 1.00f));
        colors[ImGuiCol_Border]                = LinearToSRGB(ImVec4(0.12f, 0.12f, 0.12f, 0.71f));
        colors[ImGuiCol_BorderShadow]          = LinearToSRGB(ImVec4(1.00f, 1.00f, 1.00f, 0.06f));
        colors[ImGuiCol_FrameBg]               = LinearToSRGB(ImVec4(0.42f, 0.42f, 0.42f, 0.54f));
        colors[ImGuiCol_FrameBgHovered]        = LinearToSRGB(ImVec4(0.42f, 0.42f, 0.42f, 0.40f));
        colors[ImGuiCol_FrameBgActive]         = LinearToSRGB(ImVec4(0.56f, 0.56f, 0.56f, 0.67f));
        colors[ImGuiCol_TitleBg]               = LinearToSRGB(ImVec4(0.19f, 0.19f, 0.19f, 1.00f));
        colors[ImGuiCol_TitleBgActive]         = LinearToSRGB(ImVec4(0.22f, 0.22f, 0.22f, 1.00f));
        colors[ImGuiCol_TitleBgCollapsed]      = LinearToSRGB(ImVec4(0.17f, 0.17f, 0.17f, 0.90f));
        colors[ImGuiCol_MenuBarBg]             = LinearToSRGB(ImVec4(0.335f, 0.335f, 0.335f, 1.000f));
        colors[ImGuiCol_ScrollbarBg]           = LinearToSRGB(ImVec4(0.24f, 0.24f, 0.24f, 0.53f));
        colors[ImGuiCol_ScrollbarGrab]         = LinearToSRGB(ImVec4(0.41f, 0.41f, 0.41f, 1.00f));
        colors[ImGuiCol_ScrollbarGrabHovered]  = LinearToSRGB(ImVec4(0.52f, 0.52f, 0.52f, 1.00f));
        colors[ImGuiCol_ScrollbarGrabActive]   = LinearToSRGB(ImVec4(0.76f, 0.76f, 0.76f, 1.00f));
        colors[ImGuiCol_CheckMark]             = LinearToSRGB(ImVec4(0.65f, 0.65f, 0.65f, 1.00f));
        colors[ImGuiCol_SliderGrab]            = LinearToSRGB(ImVec4(0.52f, 0.52f, 0.52f, 1.00f));
        colors[ImGuiCol_SliderGrabActive]      = LinearToSRGB(ImVec4(0.64f, 0.64f, 0.64f, 1.00f));
        colors[ImGuiCol_Button]                = LinearToSRGB(ImVec4(0.54f, 0.54f, 0.54f, 0.35f));
        colors[ImGuiCol_ButtonHovered]         = LinearToSRGB(ImVec4(0.52f, 0.52f, 0.52f, 0.59f));
        colors[ImGuiCol_ButtonActive]          = LinearToSRGB(ImVec4(0.76f, 0.76f, 0.76f, 1.00f));
        colors[ImGuiCol_Header]                = LinearToSRGB(ImVec4(0.38f, 0.38f, 0.38f, 1.00f));
        colors[ImGuiCol_HeaderHovered]         = LinearToSRGB(ImVec4(0.47f, 0.47f, 0.47f, 1.00f));
        colors[ImGuiCol_HeaderActive]          = LinearToSRGB(ImVec4(0.76f, 0.76f, 0.76f, 0.77f));
        colors[ImGuiCol_Separator]             = LinearToSRGB(ImVec4(0.000f, 0.000f, 0.000f, 0.137f));
        colors[ImGuiCol_SeparatorHovered]      = LinearToSRGB(ImVec4(0.700f, 0.671f, 0.600f, 0.290f));
        colors[ImGuiCol_SeparatorActive]       = LinearToSRGB(ImVec4(0.702f, 0.671f, 0.600f, 0.674f));
        colors[ImGuiCol_ResizeGrip]            = LinearToSRGB(ImVec4(0.26f, 0.59f, 0.98f, 0.25f));
        colors[ImGuiCol_ResizeGripHovered]     = LinearToSRGB(ImVec4(0.26f, 0.59f, 0.98f, 0.67f));
        colors[ImGuiCol_ResizeGripActive]      = LinearToSRGB(ImVec4(0.26f, 0.59f, 0.98f, 0.95f));
        colors[ImGuiCol_PlotLines]             = LinearToSRGB(ImVec4(0.61f, 0.61f, 0.61f, 1.00f));
        colors[ImGuiCol_PlotLinesHovered]      = LinearToSRGB(ImVec4(1.00f, 0.43f, 0.35f, 1.00f));
        colors[ImGuiCol_PlotHistogram]         = LinearToSRGB(ImVec4(0.90f, 0.70f, 0.00f, 1.00f));
        colors[ImGuiCol_PlotHistogramHovered]  = LinearToSRGB(ImVec4(1.00f, 0.60f, 0.00f, 1.00f));
        colors[ImGuiCol_TextSelectedBg]        = LinearToSRGB(ImVec4(0.73f, 0.73f, 0.73f, 0.35f));
        colors[ImGuiCol_ModalWindowDimBg]      = LinearToSRGB(ImVec4(0.80f, 0.80f, 0.80f, 0.35f));
        colors[ImGuiCol_DragDropTarget]        = LinearToSRGB(ImVec4(1.00f, 1.00f, 0.00f, 0.90f));
        colors[ImGuiCol_NavHighlight]          = LinearToSRGB(ImVec4(0.26f, 0.59f, 0.98f, 1.00f));
        colors[ImGuiCol_NavWindowingHighlight] = LinearToSRGB(ImVec4(1.00f, 1.00f, 1.00f, 0.70f));
        colors[ImGuiCol_NavWindowingDimBg]     = LinearToSRGB(ImVec4(0.80f, 0.80f, 0.80f, 0.20f));

        style.PopupRounding = 3;

        style.WindowPadding = ImVec2(4, 4);
        style.FramePadding  = ImVec2(6, 4);
        style.ItemSpacing   = ImVec2(6, 2);

        style.ScrollbarSize = 18;

        style.WindowBorderSize = 1;
        style.ChildBorderSize  = 1;
        style.PopupBorderSize  = 1;
        style.FrameBorderSize  = is3D;

        style.WindowRounding    = 3;
        style.ChildRounding     = 3;
        style.FrameRounding     = 3;
        style.ScrollbarRounding = 2;
        style.GrabRounding      = 3;

#ifdef IMGUI_HAS_DOCK
        style.TabBorderSize = is3D;
        style.TabRounding   = 3;

        colors[ImGuiCol_DockingEmptyBg]     = LinearToSRGB(ImVec4(0.38f, 0.38f, 0.38f, 1.00f));
        colors[ImGuiCol_Tab]                = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.25f, 1.00f));
        colors[ImGuiCol_TabHovered]         = LinearToSRGB(ImVec4(0.40f, 0.40f, 0.40f, 1.00f));
        colors[ImGuiCol_TabActive]          = LinearToSRGB(ImVec4(0.33f, 0.33f, 0.33f, 1.00f));
        colors[ImGuiCol_TabUnfocused]       = LinearToSRGB(ImVec4(0.25f, 0.25f, 0.25f, 1.00f));
        colors[ImGuiCol_TabUnfocusedActive] = LinearToSRGB(ImVec4(0.33f, 0.33f, 0.33f, 1.00f));
        colors[ImGuiCol_DockingPreview]     = LinearToSRGB(ImVec4(0.85f, 0.85f, 0.85f, 0.28f));

        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            style.WindowRounding              = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }
#endif
    }
} // namespace Ifrit::Editor::Internal
