#include "ifrit/editor/imgui/ImGuiProvider.h"
#include "ifrit.internal/editor/imgui/ImGuiUtilities.h"
#include "ifrit/runtime/renderer/profiling/ProfileDataManager.h"
#include "ifrit/runtime/base/ApplicationInterface.h"
#include <imgui.h>

namespace Ifrit::Editor::ImGuiInternal
{
    void Profiler_ShowGPUScopeStats()
    {
        auto profileDataManager = Ifrit::Runtime::GetActiveApplication()->GetProfileDataManager();
        auto gpuScopeStats      = profileDataManager->GetBriefReport();

        if (ImGui::BeginTable(
                "GPU Stats", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable))
        {
            // Setup columns
            ImGui::TableSetupColumn("Scope Name", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Avg (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Min (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableSetupColumn("Max (ms)", ImGuiTableColumnFlags_WidthFixed, 80.0f);
            ImGui::TableHeadersRow();

            // Display stats
            for (const auto& stat : gpuScopeStats)
            {
                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%s", stat.mEventName.c_str());

                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.3f", stat.mAvgDurationMs); // Convert to ms

                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%.3f", stat.mMinDurationMs);

                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%.3f", stat.mMaxDurationMs);
            }

            ImGui::EndTable();
        }
    }
} // namespace Ifrit::Editor::ImGuiInternal