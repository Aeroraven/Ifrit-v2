#include "ifrit.internal/profiler/util/RenderdocLibLoad.h"
#include "ifrit/core/hal/HalDllImport.h"
#include "ifrit/core/logging/Logging.h"
#include "renderdoc/renderdoc_app.h"
#include <filesystem>

namespace Ifrit::Profiler::Internal::Renderdoc
{
    struct RenderdocLib
    {
        HAL::FDynamicLinkedLibModule* mModule;
        RENDERDOC_API_1_6_0*          mRdocApi = NULL;
        pRENDERDOC_GetAPI             mGetApi  = NULL;
    };

    RenderdocLib& GetRenderdocLib()
    {
        static RenderdocLib renderdocLib;
        return renderdocLib;
    }

    bool LoadRenderdocLibrary()
    {
        auto& renderdocLib = GetRenderdocLib();
        // clang-format off
        Vec<String> renderdocPathCandidates = {
            "C:\\Program Files\\RenderDoc\\renderdoc.dll",
            "C:\\Program Files (x86)\\RenderDoc\\renderdoc.dll",
        };
        // clang-format on

        // chek if the path candidates are valid
        for (const auto& path : renderdocPathCandidates)
        {
            if (std::filesystem::exists(path))
            {
                renderdocLib.mModule = HAL::LoadDynamicLinkedLibrary(path.c_str());
                if (renderdocLib.mModule)
                {
                    renderdocLib.mGetApi = reinterpret_cast<pRENDERDOC_GetAPI>(
                        HAL::LoadDllFunction(renderdocLib.mModule, "RENDERDOC_GetAPI"));
                    if (renderdocLib.mGetApi)
                    {
                        if (renderdocLib.mGetApi(eRENDERDOC_API_Version_1_6_0, (void**)&renderdocLib.mRdocApi) == 1)
                        {
                            IF_LOG_INFO("Renderdoc", "Renderdoc library loaded successfully from: {}", path);
                            return true;
                        }
                        else
                        {
                            IF_LOG_ERROR("Renderdoc", "Failed to get Renderdoc API from: {}", path);
                        }
                    }
                }
                else
                {
                    IF_LOG_ERROR("Renderdoc", "Failed to load Renderdoc library from: {}", path);
                }
            }
        }
        return false;
    }

    void RequestRenderdocCaptureStart()
    {
        auto& renderdocLib = GetRenderdocLib();
        if (renderdocLib.mRdocApi)
        {
            renderdocLib.mRdocApi->StartFrameCapture(NULL, NULL);
            IF_LOG_INFO("Renderdoc", "Renderdoc capture started.");
        }
        else
        {
            IF_LOG_ERROR("Renderdoc", "Renderdoc API not initialized. Cannot start capture.");
        }
    }

    void RequestRenderdocCaptureEnd()
    {
        auto& renderdocLib = GetRenderdocLib();
        if (renderdocLib.mRdocApi)
        {
            renderdocLib.mRdocApi->EndFrameCapture(NULL, NULL);
            IF_LOG_INFO("Renderdoc", "Renderdoc capture ended.");
        }
        else
        {
            IF_LOG_ERROR("Renderdoc", "Renderdoc API not initialized. Cannot end capture.");
        }
    }
} // namespace Ifrit::Profiler::Internal::Renderdoc