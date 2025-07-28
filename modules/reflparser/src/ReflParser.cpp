// Ifrit Reflection Parser - Refactored
#include "clang-c/Index.h"
#include <vector>

#include <string>

#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>

#include <filesystem>
#include <stdexcept>
#include <cctype>
#include <cstring>

#include "ifrit.internal/reflparse/ReflParseLog.h"
#include "ifrit.internal/reflparse/PropertyParser.h"
namespace Ifrit::ReflParser
{

    // --- Types and Context ---
    enum class RecordType
    {
        Class,
        Property,
        IncludeFiles,
        PolymorphicRelation,
    };

    struct Record
    {
        RecordType            type;
        std::string           symbolName;
        std::string           propertyAlias;
        std::unique_ptr<Node> node;
    };

    struct ReflectionParserContext
    {
        std::vector<Record>             records;
        std::unordered_set<std::string> registeredTypes;
        std::vector<std::string>        listHeaders;
        bool                            shouldAddHeader = false;
    };

    // --- Utility Functions ---
    inline bool fileExists(const char* filename)
    {
        std::ifstream file(filename);
        return file.good();
    }

    std::string splitCamelCase(const std::string& input)
    {
        if (input.empty())
            return input;
        std::string result;
        // NOTE: ALWAYS SKIP THE FIRST
        for (size_t i = 1; i < input.length(); ++i)
        {
            char current = input[i];
            if (i > 0 && std::isupper(current))
            {

                if (!std::isupper(input[i - 1]) && !result.empty())
                    result += " ";
                else if (i < input.length() - 1 && std::islower(input[i + 1]) && !result.empty())
                    result += " ";
            }
            result += current;
        }
        return result;
    }

    // --- Clang Annotation Helpers ---
    bool hasIfritReflClassAnnotation(CXCursor cursor)
    {
        bool hasAnnotation = false;
        clang_visitChildren(
            cursor,
            [](CXCursor child, CXCursor, CXClientData data) -> CXChildVisitResult {
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

    bool hasIfritReflPropertyAnnotation(CXCursor cursor, Node& node)
    {

        struct NodeVisitorData
        {
            Node* node;
            bool* found;
        } clientData;

        bool hasAnnotation = false;
        clientData.node    = &node;
        clientData.found   = &hasAnnotation;
        clang_visitChildren(
            cursor,
            [](CXCursor child, CXCursor, CXClientData data) -> CXChildVisitResult {
                NodeVisitorData* clientData = static_cast<NodeVisitorData*>(data);
                bool*            found      = clientData->found;
                Node*            node       = clientData->node;
                if (clang_getCursorKind(child) == CXCursor_AnnotateAttr)
                {
                    CXString    annotation    = clang_getCursorSpelling(child);
                    const char* annotationStr = clang_getCString(annotation);
                    if (annotationStr && std::string(annotationStr).starts_with("ifrit.refl.property:"))
                    {
                        std::string annotationContent =
                            std::string(annotationStr).substr(std::string("ifrit.refl.property:").length());
                        *node = PropParse::parse("(" + annotationContent + ")");

                        *found = true;
                        clang_disposeString(annotation);
                        return CXChildVisit_Break;
                    }
                    clang_disposeString(annotation);
                }
                return CXChildVisit_Continue;
            },
            &clientData);
        return hasAnnotation;
    }

    std::string getFullyQualifiedName(CXCursor cursor)
    {

        std::vector<std::string> nameParts;
        CXString                 name    = clang_getCursorSpelling(cursor);
        const char*              nameStr = clang_getCString(name);
        if (nameStr && strlen(nameStr) > 0)
            nameParts.push_back(std::string(nameStr));
        clang_disposeString(name);
        CXCursor parent = clang_getCursorSemanticParent(cursor);
        while (!clang_Cursor_isNull(parent)
            && !clang_equalCursors(parent, clang_getTranslationUnitCursor(clang_Cursor_getTranslationUnit(cursor))))
        {

            CXCursorKind parentKind = clang_getCursorKind(parent);
            if (parentKind == CXCursor_Namespace || parentKind == CXCursor_ClassDecl
                || parentKind == CXCursor_StructDecl)
            {

                CXString    parentName    = clang_getCursorSpelling(parent);
                const char* parentNameStr = clang_getCString(parentName);
                if (parentNameStr && strlen(parentNameStr) > 0)
                    nameParts.insert(nameParts.begin(), std::string(parentNameStr));
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

    // --- Reflection Parsing Logic ---
    void printAnnotatedFields(
        CXCursor classCursor, const std::string& fullyQualifiedClassName, ReflectionParserContext& ctx)
    {

        clang_visitChildren(
            classCursor,
            [](CXCursor cursor, CXCursor, CXClientData client_data) -> CXChildVisitResult {
                auto* params = static_cast<std::pair<const std::string*, ReflectionParserContext*>*>(client_data);
                const std::string&       className = *params->first;
                ReflectionParserContext& ctx       = *params->second;
                if (clang_getCursorKind(cursor) == CXCursor_FieldDecl)
                {
                    std::unique_ptr<Node> node = std::make_unique<Node>();
                    if (hasIfritReflPropertyAnnotation(cursor, *node))
                    {
                        CXString    fieldName         = clang_getCursorSpelling(cursor);
                        const char* fieldNameStr      = clang_getCString(fieldName);
                        std::string fieldNameString   = fieldNameStr ? fieldNameStr : "<anonymous>";
                        std::string humanReadableName = splitCamelCase(fieldNameString);
                        LogInfo("Field: ", className.c_str(), "::", fieldNameString.c_str(), "(",
                            humanReadableName.c_str(), ")");
                        ctx.records.push_back({ RecordType::Property, className + "::" + fieldNameString,
                            humanReadableName, std::move(node) });
                        clang_disposeString(fieldName);
                    }
                }
                return CXChildVisit_Continue;
            },
            new std::pair<const std::string*, ReflectionParserContext*>(&fullyQualifiedClassName, &ctx));
    }

    std::vector<std::string> getBaseClasses(CXCursor classCursor)
    {
        std::vector<std::string> baseClasses;
        clang_visitChildren(
            classCursor,
            [](CXCursor cursor, CXCursor, CXClientData client_data) -> CXChildVisitResult {
                auto* baseClassesVector = static_cast<std::vector<std::string>*>(client_data);
                if (clang_getCursorKind(cursor) == CXCursor_CXXBaseSpecifier)
                {
                    CXCursor baseClassCursor = clang_getCursorReferenced(cursor);
                    if (!clang_Cursor_isNull(baseClassCursor))
                    {
                        std::string baseClassName = getFullyQualifiedName(baseClassCursor);
                        if (!baseClassName.empty())
                            baseClassesVector->push_back(baseClassName);
                    }
                }
                return CXChildVisit_Continue;
            },
            &baseClasses);
        return baseClasses;
    }

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

    int RecursiveParse(const std::string& path, ReflectionParserContext& ctx)
    {

        collectHeaders(path, ctx.listHeaders);
        std::ofstream outHeads(IFRIT_GATHER_HEADS_OUT);
        for (auto& p : ctx.listHeaders)
        {

            std::string   fileContent;
            std::ifstream file(p);
            if (file)
            {
                fileContent = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
                file.close();
            }
            if (fileContent.find("IF_CLASS") == std::string::npos && fileContent.find("IF_STRUCT") == std::string::npos)
            {
                continue;
            }
            LogInfo("Adding header: ", p.c_str());
            outHeads << "#include \"" << p << "\"" << std::endl;
            ctx.records.push_back({ RecordType::IncludeFiles, p, "" });
        }
        outHeads.close();
        return 0;
    }

    int ParseFile(const char* subdir, ReflectionParserContext& ctx)
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
        auto                     index     = clang_createIndex(0, 1);
        CXTranslationUnit        unit      = nullptr;
        auto                     result =
            clang_parseTranslationUnit2(index, targetFile, arguments.data(), (int)arguments.size(), nullptr, 0,
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
            CXCursor cursor     = clang_getTranslationUnitCursor(unit);
            ctx.shouldAddHeader = false;
            clang_visitChildren(
                cursor,
                [](CXCursor cursor, CXCursor, CXClientData client_data) -> CXChildVisitResult {
                    ReflectionParserContext& ctx  = *static_cast<ReflectionParserContext*>(client_data);
                    CXCursorKind             kind = clang_getCursorKind(cursor);
                    if (kind == CXCursor_StructDecl || kind == CXCursor_ClassDecl)
                    {

                        bool              hasIfClass = false;
                        CXToken*          tokens;
                        unsigned          numTokens;
                        CXSourceRange     range = clang_getCursorExtent(cursor);
                        CXTranslationUnit tu    = clang_Cursor_getTranslationUnit(cursor);
                        clang_tokenize(tu, range, &tokens, &numTokens);
                        for (unsigned i = 0; i < numTokens; ++i)
                        {

                            CXString    tokenString = clang_getTokenSpelling(tu, tokens[i]);
                            const char* tokenText   = clang_getCString(tokenString);
                            if (tokenText
                                && (strcmp(tokenText, "IF_CLASS") == 0 || strcmp(tokenText, "IF_STRUCT") == 0))
                            {
                                hasIfClass = true;
                            }
                            clang_disposeString(tokenString);
                            if (hasIfClass)
                                break;
                        }
                        clang_disposeTokens(tu, tokens, numTokens);
                        if (hasIfClass || hasIfritReflClassAnnotation(cursor))
                        {
                            std::string fullyQualifiedName = getFullyQualifiedName(cursor);
                            if (fullyQualifiedName.empty())
                                fullyQualifiedName = "<anonymous>";
                            LogInfo("Found annotated class/struct: ", fullyQualifiedName.c_str());
                            ctx.registeredTypes.insert(fullyQualifiedName);
                            std::vector<std::string> baseClasses = getBaseClasses(cursor);
                            ctx.records.push_back({ RecordType::Class, fullyQualifiedName, "" });
                            if (!baseClasses.empty())
                            {
                                LogInfo("Base classes for ", fullyQualifiedName.c_str(), ":");
                                for (const auto& baseClass : baseClasses)
                                {
                                    LogInfo("  - ", baseClass.c_str());
                                    if (ctx.registeredTypes.find(baseClass) == ctx.registeredTypes.end())
                                    {
                                        LogInfo("Base class ", baseClass.c_str(), " is not registered, skipping.");
                                        continue;
                                    }
                                    ctx.records.push_back(
                                        { RecordType::PolymorphicRelation, fullyQualifiedName, baseClass });
                                }
                            }
                            else
                            {
                                LogInfo("No base classes for ", fullyQualifiedName.c_str());
                            }
                            ctx.shouldAddHeader = true;
                            printAnnotatedFields(cursor, fullyQualifiedName, ctx);
                        }
                    }

                    return CXChildVisit_Recurse;
                },
                &ctx);
            clang_disposeTranslationUnit(unit);
        }
        clang_disposeIndex(index);
        return 0;
    }

    void GenerateCode(const std::string& outdir, const ReflectionParserContext& ctx)
    {
        std::stringstream outputStream;
        outputStream << "// ****************************************************************\n";
        outputStream << "// This file is auto-generated by ifrit.reflparser.\n";
        outputStream << "// Do not edit this file manually.\n";
        outputStream << "// ****************************************************************\n\n";
        outputStream << "#pragma once\n";
        outputStream << "#include \"ifrit/core/reflection/Reflection.h\"\n\n";
        outputStream << "// Begin Body\n";
        for (const auto& record : ctx.records)
        {
            if (record.type == RecordType::IncludeFiles)
            {
                outputStream << "#include \"" << record.symbolName << "\"\n";
            }
        }
        outputStream << "namespace Ifrit::Reflection\n{\n";
        outputStream << "    void RegisterReflectionTypes()\n    {\n";
        for (const auto& record : ctx.records)
        {
            if (record.type == RecordType::Class)
            {
                outputStream << "\n        // " << record.symbolName << "\n";
                outputStream << "        RegisterType<" << record.symbolName << ">();\n";
            }
            else if (record.type == RecordType::Property)
            {
                outputStream << "        RegisterPropertyField<&" << record.symbolName << ">(\"" << record.propertyAlias
                             << "\");\n";
                if (record.node)
                    PropParse::printNode(*record.node, outputStream, 4);
            }
            else if (record.type == RecordType::PolymorphicRelation)
            {
                outputStream << "        RegisterPolymorphicRelation<" << record.symbolName << ", "
                             << record.propertyAlias << ">();\n";
            }
        }
        outputStream << "    }\n";
        outputStream << "}\n";
        std::ofstream outFile(outdir);
        outFile << outputStream.str();
        outFile.close();
    }

} // namespace Ifrit::ReflParser

// --- Main ---
int main(int argc, char** argv)
{

    using namespace Ifrit::ReflParser;
    ReflectionParserContext  ctx;
    std::vector<std::string> inputPaths;
    std::string              outputFile;
    for (int i = 1; i < argc; ++i)
    {

        if (strcmp(argv[i], "--input") == 0)
        {

            while (++i < argc)
            {
                if (argv[i][0] == '-')
                {
                    --i;
                    break;
                }
                inputPaths.push_back(argv[i]);
            }
        }
        else if (strcmp(argv[i], "--output") == 0)
        {
            if (++i < argc)
            {
                outputFile = argv[i];
            }
            else
            {
                std::cerr << "ERROR: No output file specified after --output" << std::endl;
                return 1;
            }
        }

        else
        {

            std::cerr << "ERROR: Unknown argument: " << argv[i] << std::endl;
            return 1;
        }
    }

    RecursiveParse(IFRIT_COMMON_INCLUDE_DIR "/ifrit/runtime", ctx);
    for (const auto& path : inputPaths)
    {

        RecursiveParse(path, ctx);
    }

    ParseFile(IFRIT_GATHER_HEADS_OUT, ctx);
    if (!std::filesystem::exists(std::filesystem::path(outputFile).parent_path()))
    {
        std::filesystem::create_directories(std::filesystem::path(outputFile).parent_path());
        std::cout << "Created output directory: " << std::filesystem::path(outputFile).parent_path() << std::endl;
    }
    GenerateCode(outputFile, ctx);
    return 0;
}