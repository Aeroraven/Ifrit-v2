#pragma once
#include "ifrit/core/typing/Traits.h"
#include "ifrit/core/base/containers/Maps.h"
#include "ifrit/core/base/CoreBase.h"

namespace Ifrit
{
    enum class EGraphVisualizeNodeShape
    {
        Box,
        Circle,
    };

    class Graph
    {
    public:
        struct Node
        {
            String                   mIdentifier;
            THashMap<String, String> mAttributes;
            Vec<String>              mOutEdges;
            EGraphVisualizeNodeShape mShape = EGraphVisualizeNodeShape::Box;
        };

        Graph()  = default;
        ~Graph() = default;

        void AddNode(const String& name, const THashMap<String, String>& attributes = {})
        {
            mNodes.emplace(name, Node{ .mIdentifier = name, .mAttributes = attributes });
        }
        void AddEdge(const String& from, const String& to)
        {
            auto it = mNodes.find(from);
            if (it != mNodes.end())
            {
                it->second.mOutEdges.push_back(to);
            }
        }
        void AddAttribute(const String& node, const String& key, const String& value)
        {
            auto it = mNodes.find(node);
            if (it != mNodes.end())
            {
                it->second.mAttributes[key] = value;
            }
        }
        void SetNodeShape(const String& node, EGraphVisualizeNodeShape shape)
        {
            auto it = mNodes.find(node);
            if (it != mNodes.end())
            {
                it->second.mShape = shape;
            }
        }

        IFRIT_CORE_API String ToDotString() const;

    private:
        THashMap<String, Node> mNodes;
    };
} // namespace Ifrit