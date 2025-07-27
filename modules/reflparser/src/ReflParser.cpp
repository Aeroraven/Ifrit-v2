#include "clang-c/Index.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <sstream>
enum class RecordType
{
    Class,
    Property,
    IncludeFiles, // #include
};

struct Records
{
    RecordType  type;
    std::string symbolName;
    std::string propertyAlias;
};

std::vector<Records>     gRecords;
std::vector<std::string> listHeaders;

bool                     fileExists(const char* filename)
{
    std::ifstream file(filename);
    return file.good();
}

bool hasIfritReflClassAnnotation(CXCursor cursor)
{
    bool hasAnnotation = false;
    clang_visitChildren(
        cursor,
        [](CXCursor child, CXCursor parent, CXClientData data) -> CXChildVisitResult {
            bool* found = static_cast<bool*>(data);
            if (clang_getCursorKind(child) == CXCursor_AnnotateAttr)
            {
                CXString    annotation    = clang_getCursorSpelling(child);
                const char* annotationStr = clang_getCString(annotation);
                if (annotationStr && std::string(annotationStr) == "ifrit.refl.class")
                {
                    *found = true;
                    clang_disposeString(annotation);
                    return CXChildVisit_Break;
                }
                clang_disposeString(annotation);
            }
            return CXChildVisit_Continue;
        },
        &hasAnnotation);

    return hasAnnotation;
}

bool hasIfritReflPropertyAnnotation(CXCursor cursor)
{
    bool hasAnnotation = false;
    clang_visitChildren(
        cursor,
        [](CXCursor child, CXCursor parent, CXClientData data) -> CXChildVisitResult {
            bool* found = static_cast<bool*>(data);

            if (clang_getCursorKind(child) == CXCursor_AnnotateAttr)
            {
                CXString    annotation    = clang_getCursorSpelling(child);
                const char* annotationStr = clang_getCString(annotation);

                if (annotationStr && std::string(annotationStr) == "ifrit.refl.property")
                {
                    *found = true;
                    clang_disposeString(annotation);
                    return CXChildVisit_Break;
                }
                clang_disposeString(annotation);
            }

            return CXChildVisit_Continue;
        },
        &hasAnnotation);

    return hasAnnotation;
}

std::string getFullyQualifiedName(CXCursor cursor)
{
    std::vector<std::string> nameParts;
    CXString                 name    = clang_getCursorSpelling(cursor);
    const char*              nameStr = clang_getCString(name);
    if (nameStr && strlen(nameStr) > 0)
    {
        nameParts.push_back(std::string(nameStr));
    }
    clang_disposeString(name);
    CXCursor parent = clang_getCursorSemanticParent(cursor);
    while (!clang_Cursor_isNull(parent)
        && !clang_equalCursors(parent, clang_getTranslationUnitCursor(clang_Cursor_getTranslationUnit(cursor))))
    {
        CXCursorKind parentKind = clang_getCursorKind(parent);
        if (parentKind == CXCursor_Namespace || parentKind == CXCursor_ClassDecl || parentKind == CXCursor_StructDecl)
        {
            CXString    parentName    = clang_getCursorSpelling(parent);
            const char* parentNameStr = clang_getCString(parentName);
            if (parentNameStr && strlen(parentNameStr) > 0)
            {
                nameParts.insert(nameParts.begin(), std::string(parentNameStr));
            }
            clang_disposeString(parentName);
        }

        parent = clang_getCursorSemanticParent(parent);
    }

    std::string result;
    for (size_t i = 0; i < nameParts.size(); ++i)
    {
        if (i > 0)
            result += "::";
        result += nameParts[i];
    }

    return result;
}

std::string splitCamelCase(const std::string& input)
{
    if (input.empty())
        return input;

    std::string result;

    for (size_t i = 0; i < input.length(); ++i)
    {
        char current = input[i];
        if (i > 0 && std::isupper(current))
        {
            if (i > 0 && !std::isupper(input[i - 1]))
            {
                result += " ";
            }
            else if (i < input.length() - 1 && std::islower(input[i + 1]))
            {
                result += " ";
            }
        }
        if (i == 0)
        {
            result += std::toupper(current);
        }
        else
        {
            result += current;
        }
    }

    return result;
}

void printAnnotatedFields(CXCursor classCursor, const std::string& fullyQualifiedClassName)
{
    clang_visitChildren(
        classCursor,
        [](CXCursor cursor, CXCursor parent, CXClientData client_data) -> CXChildVisitResult {
            const std::string* className = static_cast<const std::string*>(client_data);
            CXCursorKind       kind      = clang_getCursorKind(cursor);

            if (kind == CXCursor_FieldDecl)
            {

                if (hasIfritReflPropertyAnnotation(cursor))
                {
                    CXString    fieldName    = clang_getCursorSpelling(cursor);
                    const char* fieldNameStr = clang_getCString(fieldName);

                    std::string fieldNameString   = fieldNameStr ? fieldNameStr : "<anonymous>";
                    std::string humanReadableName = splitCamelCase(fieldNameString);

                    std::cout << "  Field: &" << *className << "::" << fieldNameString << " (" << humanReadableName
                              << ")" << std::endl;

                    gRecords.push_back(
                        { RecordType::Property, (*className) + "::" + fieldNameString, humanReadableName });

                    clang_disposeString(fieldName);
                }
            }

            return CXChildVisit_Continue;
        },
        const_cast<std::string*>(&fullyQualifiedClassName));
}

void GenerateCode()
{
    // RegisterType<Class>()
    // RegisterPropertyField<&Class::field>("alias");
    std::stringstream outputStream;
    outputStream << "// ****************************************************************\n";
    outputStream << "// This file is auto-generated by ifrit.reflparser.\n";
    outputStream << "// Do not edit this file manually.\n";
    outputStream << "// ****************************************************************\n\n";

    outputStream << "#include \"ifrit/core/reflection/Reflection.h\"\n\n";

    outputStream << "// Begin Body\n";
    // headers
    for (const auto& record : gRecords)
    {
        if (record.type == RecordType::IncludeFiles)
        {
            outputStream << "#include \"" << record.symbolName << "\"\n";
        }
    }
    outputStream << "namespace Ifrit::Reflection\n{\n";
    outputStream << "    void RegisterReflectionTypes()\n    {\n";
    for (const auto& record : gRecords)
    {
        if (record.type == RecordType::Class)
        {
            outputStream << "        RegisterType<" << record.symbolName << ">();\n";
        }
        else if (record.type == RecordType::Property)
        {
            outputStream << "        RegisterPropertyField<&" << record.symbolName << ">(\"" << record.propertyAlias
                         << "\");\n";
        }
    }
    outputStream << "    }\n";
    outputStream << "}\n";

    // print to stdout
    std::ofstream outFile(IFRIT_META_OUT);
    outFile << outputStream.str();
    outFile.close();
}

bool gShouldAddHeader = false;
int  ParseFile(const char* subdir)
{
    std::string targetFileT = subdir;
    const char* targetFile  = targetFileT.c_str();

    if (!fileExists(targetFile))
    {
        std::cerr << "ERROR: File does not exist: " << targetFile << std::endl;
        return 1;
    }

    std::vector<const char*> arguments = { "-std=c++20", "-Wno-unknown-attributes", "-fparse-all-comments",
        "-D__clang__", "-DIF_REFLPARSER_PASS", "-I", IFRIT_COMMON_INCLUDE_DIR, "-I", IFRIT_COMMON_DEPS_INCLUDE_DIR,
        "-x", "c++-header", "-w" };

    auto                     index = clang_createIndex(0, 1);
    CXTranslationUnit        unit  = nullptr;
    auto result = clang_parseTranslationUnit2(index, targetFile, arguments.data(), (int)arguments.size(), nullptr, 0,
        CXTranslationUnit_None | CXTranslationUnit_SkipFunctionBodies
            | CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles | CXTranslationUnit_KeepGoing,
        &unit);

    if (result != CXError_Success)
    {
        std::cerr << "\nFailed to parse translation unit. Error code: " << result << std::endl;
        return 1;
    }
    if (unit)
    {
        CXCursor cursor = clang_getTranslationUnitCursor(unit);

        gShouldAddHeader = false;
        clang_visitChildren(
            cursor,
            [](CXCursor cursor, CXCursor parent, CXClientData client_data) -> CXChildVisitResult {
                CXCursorKind kind = clang_getCursorKind(cursor);

                if (kind == CXCursor_StructDecl || kind == CXCursor_ClassDecl)
                {

                    if (hasIfritReflClassAnnotation(cursor))
                    {

                        std::string fullyQualifiedName = getFullyQualifiedName(cursor);

                        if (fullyQualifiedName.empty())
                        {
                            fullyQualifiedName = "<anonymous>";
                        }

                        std::cout << "Found annotated class/struct: " << fullyQualifiedName << std::endl;
                        gRecords.push_back({ RecordType::Class, fullyQualifiedName, "" });
                        gShouldAddHeader = true;
                        printAnnotatedFields(cursor, fullyQualifiedName);
                    }
                }

                return CXChildVisit_Recurse;
            },
            nullptr);

        clang_disposeTranslationUnit(unit);
    }

    clang_disposeIndex(index);
    return 0;
}

// Helper function to collect headers recursively (without file writing)
void collectHeaders(const std::string& path, std::vector<std::string>& listHeaders)
{
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        if (entry.is_directory())
        {
            collectHeaders(entry.path().string(), listHeaders);
        }
        else if (entry.is_regular_file() && entry.path().extension() == ".h")
        {
            listHeaders.push_back(entry.path().string());
        }
    }
}

int RecursiveParse(const std::string& path)
{

    // Collect all headers recursively
    collectHeaders(path, listHeaders);

    // Write all headers to file once
    std::ofstream outHeads(IFRIT_GATHER_HEADS_OUT);
    for (auto& p : listHeaders)
    {
        std::cout << "ADD HEADER:" << p << "\n";
        outHeads << "#include \"" << p << "\"" << std::endl;
        gRecords.push_back({ RecordType::IncludeFiles, p, "" });
    }
    outHeads.close();

    return 0;
}

int main()
{
    RecursiveParse(IFRIT_COMMON_INCLUDE_DIR "/ifrit/runtime/base");
    ParseFile(IFRIT_GATHER_HEADS_OUT);
    GenerateCode();
    return 0;
}