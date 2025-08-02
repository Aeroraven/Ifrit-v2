#include "ifrit/editor/widgets/FileDialog.h"
#include "nfd/src/include/nfd.hpp"
#include <codecvt>
#include <locale>

#ifdef _WIN32
    #define NFD_NATIVE_WCHAR
#endif

namespace Ifrit::Editor::Widget
{
    // Helper function to convert string to wide string
    std::wstring stringToWstring(const std::string& str)
    {
        if (str.empty())
            return std::wstring();
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.from_bytes(str);
    }

    // Helper function to convert wide string to string
    std::string wstringToString(const std::wstring& wstr)
    {
        if (wstr.empty())
            return std::string();
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        return converter.to_bytes(wstr);
    }

    void              FileDialog::Initialize() { NFD::Init(); }

    void              FileDialog::Finalize() { NFD::Quit(); }

    FFileDialogResult FileDialog::OpenFileDialog(const FFileDialogSetupArgs& args)
    {
        nfdnchar_t*           outPath = nullptr;
        Vec<nfdnfilteritem_t> filterItems;

        // Convert filter items if needed
        Vec<std::wstring>     wideNames;
        Vec<std::wstring>     wideSpecs;

        for (const auto& fileType : args.mFileTypes)
        {
            nfdnfilteritem_t item;

#ifdef NFD_NATIVE_WCHAR
            // If NFD uses wide characters, convert strings
            wideNames.push_back(stringToWstring(fileType.first));
            wideSpecs.push_back(stringToWstring(fileType.second));
            item.name = wideNames.back().c_str();
            item.spec = wideSpecs.back().c_str();
#else
            // If NFD uses narrow characters, use directly
            item.name = fileType.first.c_str();
            item.spec = fileType.second.c_str();
#endif

            filterItems.push_back(item);
        }

        nfdresult_t       result = NFD::SaveDialog(outPath, filterItems.data(), filterItems.size(), nullptr);

        FFileDialogResult dialogResult;
        if (result == NFD_OKAY)
        {
            dialogResult.mSuccess = true;

#ifdef NFD_NATIVE_WCHAR
            // Convert wide string back to narrow string
            std::wstring widePath(reinterpret_cast<const wchar_t*>(outPath));
            dialogResult.mFilePath = wstringToString(widePath);
#else
            // Use narrow string directly
            dialogResult.mFilePath = outPath;
#endif

            NFD::FreePath(outPath);
        }
        else
        {
            dialogResult.mSuccess = false;
            dialogResult.mFilePath.clear();
        }
        return dialogResult;
    }
} // namespace Ifrit::Editor::Widget