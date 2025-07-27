#include "clang-c/Index.h"
#include <vector>
#include <stdexcept>
#include <iostream>

int main()
{

    std::vector<const char*> arguments = { "-std=c++17", "-D__clang__", "-D__META_PARSER__" };

    auto                     index = clang_createIndex(0, 0);

    CXTranslationUnit        unit;

    auto result = clang_parseTranslationUnit2(index, "C:/WR/Ifrit-v2/modules/reflparser/dummy/Test.cpp",
        arguments.data(), (int)arguments.size(), nullptr, 0, CXTranslationUnit_None, &unit);

    if (result != CXError_Success)
    {
        std::cerr << "Failed to parse translation unit. Error code: " << result << std::endl;
        clang_disposeIndex(index);
        return 1;
    }

    std::cout << "Translation unit parsed successfully!" << std::endl;

    CXCursor cursor = clang_getTranslationUnitCursor(unit);
    clang_visitChildren(
        cursor,
        [](CXCursor cursor, CXCursor parent, CXClientData client_data) -> CXChildVisitResult {
            CXString     name     = clang_getCursorSpelling(cursor);
            CXCursorKind kind     = clang_getCursorKind(cursor);
            CXString     kindName = clang_getCursorKindSpelling(kind);

            std::cout << "Found: " << clang_getCString(name) << " (kind: " << clang_getCString(kindName) << ")"
                      << std::endl;

            clang_disposeString(name);
            clang_disposeString(kindName);

            return CXChildVisit_Recurse;
        },
        nullptr);

    clang_disposeTranslationUnit(unit);
    clang_disposeIndex(index);
    return 0;
}