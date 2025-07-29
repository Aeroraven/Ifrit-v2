#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <sstream>
#include <stdexcept>
#include "ifrit.internal/reflparse/ReflParseLog.h"

template <typename... Args> void LogInfo(Args&&... args);

struct Node
{
    std::string                                               key;
    std::variant<std::string, double, int, std::vector<Node>> value;
};

namespace Ifrit::ReflParser::PropParse
{
    void skipWhitespace(const std::string& str, size_t& pos)
    {
        while (pos < str.size() && isspace(str[pos]))
        {
            ++pos;
        }
    }

    std::string parseString(const std::string& str, size_t& pos)
    {
        if (str[pos] == '\'' || str[pos] == '"')
        {
            char        quote = str[pos++];
            std::string result;
            while (pos < str.size() && str[pos] != quote)
            {
                result += str[pos++];
            }
            if (pos >= str.size() || str[pos] != quote)
            {
                LogInfo("Error: Unmatched quote in string");
                throw std::runtime_error("Unmatched quote in string");
            }
            ++pos;
            return result;
        }
        LogInfo("Error: Expected string to start with ' or \"");
        throw std::runtime_error("Invalid string format");
    }

    std::variant<double, int> parseNumber(const std::string& str, size_t& pos)
    {
        size_t start = pos;
        while (pos < str.size() && (isdigit(str[pos]) || str[pos] == '.' || str[pos] == 'f'))
        {
            ++pos;
        }
        std::string numStr = str.substr(start, pos - start);
        if (numStr.find('.') != std::string::npos || numStr.back() == 'f')
        {
            return std::stod(numStr);
        }
        return std::stoi(numStr);
    }

    std::vector<Node> parseObject(const std::string& str, size_t& pos);

    Node              parseKeyValue(const std::string& str, size_t& pos)
    {
        skipWhitespace(str, pos);

        size_t start = pos;
        while (pos < str.size() && (isalnum(str[pos]) || str[pos] == '_'))
        {
            ++pos;
        }
        if (start == pos)
        {
            LogInfo("Error: Invalid key format at position ", start);
            throw std::runtime_error("Invalid key format");
        }
        std::string key = str.substr(start, pos - start);

        skipWhitespace(str, pos);

        if (pos < str.size() && str[pos] == '=')
        {
            ++pos;
            skipWhitespace(str, pos);

            if (str[pos] == '\'' || str[pos] == '"')
            {
                return { key, parseString(str, pos) };
            }
            else if (isdigit(str[pos]) || str[pos] == '-')
            {
                auto number = parseNumber(str, pos);

                if (std::holds_alternative<double>(number))
                {
                    return { key, std::get<double>(number) };
                }
                else
                {
                    return { key, std::get<int>(number) };
                }
            }
            else if (str[pos] == '(')
            {
                ++pos;
                return { key, parseObject(str, pos) };
            }
            else
            {
                LogInfo("Error: Invalid value format for key '", key, "'");
                throw std::runtime_error("Invalid value format");
            }
        }

        return { key, "" };
    }

    std::vector<Node> parseObject(const std::string& str, size_t& pos)
    {
        std::vector<Node> nodes;
        while (pos < str.size())
        {
            skipWhitespace(str, pos);

            if (str[pos] == ')')
            {
                ++pos;
                break;
            }

            nodes.push_back(parseKeyValue(str, pos));

            skipWhitespace(str, pos);

            if (pos < str.size() && str[pos] == ',')
            {
                ++pos;
            }
        }
        return nodes;
    }

    Node parse(const std::string& str)
    {
        size_t pos = 0;
        skipWhitespace(str, pos);
        LogInfo("Parsing input: ", str);
        if (str[pos] == '(')
        {
            ++pos;
            return { "root", parseObject(str, pos) };
        }
        LogInfo("Error: Expected '(' at the start of the input");
        throw std::runtime_error("Invalid input format");
    }

    void printNode(const Node& node, const std::string& memberName, std::stringstream& stream,
        const std::string& parentKey = "", int indent = 0)
    {
        std::string indentStr(indent, ' ');
        std::string fullKey = parentKey.empty() ? node.key : parentKey + "." + node.key;
        if (indent == 0)
            fullKey = "";

        // LogInfo for debugging

        if (std::holds_alternative<std::string>(node.value))
        {
            LogInfo(fullKey, ":", std::get<std::string>(node.value));
            stream << "        ";
            stream << "RegisterPropertyHint<&" << memberName << ">(\"" << fullKey << "\", \""
                   << std::get<std::string>(node.value) << "\");\n";
        }
        else if (std::holds_alternative<double>(node.value))
        {
            LogInfo(fullKey, ":", std::get<double>(node.value));
            stream << "        ";
            stream << "RegisterPropertyHint<&" << memberName << ">(\"" << fullKey << "\", (double)"
                   << std::get<double>(node.value) << ");\n";
        }
        else if (std::holds_alternative<int>(node.value))
        {
            LogInfo(fullKey, ":", std::get<int>(node.value));
            stream << "        ";
            stream << "RegisterPropertyHint<&" << memberName << ">(\"" << fullKey << "\", (int)"
                   << std::get<int>(node.value) << ");\n";
        }
        else if (std::holds_alternative<std::vector<Node>>(node.value))
        {
            for (const auto& child : std::get<std::vector<Node>>(node.value))
            {
                printNode(child, memberName, stream, fullKey, indent + 2);
            }
        }
    }
} // namespace Ifrit::ReflParser::PropParse
  //"Flag1,Flag2=(Prop1='va1',Prop2=1.0f,Prop3=\"www\"),Flag3=(FlagInner1,Prop3Inner=(Prop1='a',Prop2='b'))";