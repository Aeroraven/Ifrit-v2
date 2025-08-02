#pragma once

#include "ifrit/core/base/IfritBase.h"
#include "ifrit/editor/EditorBase.h"

namespace Ifrit::Editor::Widget
{
    enum class EFileDialogType
    {
        OpenFile,
        SaveFile,
    };

    struct FFileDialogResult
    {
        bool   mSuccess = false;
        String mFilePath;
    };

    struct FFileDialogSetupArgs
    {
        EFileDialogType           mDialogType = EFileDialogType::SaveFile;
        Vec<Pair<String, String>> mFileTypes;
    };

    class IFRIT_EDITOR_API FileDialog
    {
    public:
        void              Initialize();
        void              Finalize();
        FFileDialogResult OpenFileDialog(const FFileDialogSetupArgs& args);
    };
} // namespace Ifrit::Editor::Widget
