#pragma once
#include "./core/definition/CoreExports.h"
#include "./engine/base/VertexBuffer.h"

namespace Ifrit::Engine::MeshletBuilder {
    struct Meshlet{
        VertexBuffer vbufs;
        std::vector<int> ibufs;
    };
};