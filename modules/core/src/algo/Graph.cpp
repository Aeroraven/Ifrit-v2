#include "ifrit/core/algo/Graph.h"

namespace Ifrit
{
    // ===== Graph Impls =====

    IFRIT_CORE_API String Graph::ToDotString() const
    {
        String result = "digraph G {\n";
        result += "    rankdir=TB;\n";
        result += "    node [fontname=\"Arial\", fontsize=10];\n";
        result += "    edge [fontname=\"Arial\", fontsize=8];\n\n";

        // Add nodes with attributes
        for (const auto& [nodeName, node] : mNodes)
        {
            result += "    \"" + nodeName + "\" [";

            // Add shape attribute
            if (node.mShape == EGraphVisualizeNodeShape::Circle)
            {
                result += "shape=circle";
            }
            else
            {
                result += "shape=box";
            }

            // Add label attribute (required)
            result += ", label=\"" + nodeName + "\"";

            // Add custom attributes
            for (const auto& [key, value] : node.mAttributes)
            {
                result += ", " + key + "=\"" + value + "\"";
            }

            result += "];\n";
        }

        result += "\n";

        // Add edges
        for (const auto& [fromNode, node] : mNodes)
        {
            for (const auto& toNode : node.mOutEdges)
            {
                result += "    \"" + fromNode + "\" -> \"" + toNode + "\";\n";
            }
        }

        result += "}\n";
        return result;
    }
} // namespace Ifrit