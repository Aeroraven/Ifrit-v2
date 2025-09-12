#pragma once

#include "ifrit/core/logging/Logging.h"

namespace Ifrit::Runtime::RenderCore::RDG
{
#define RDG_LOG_TRACE(...) IF_LOG_TRACE("RenderGraph", __VA_ARGS__)
#define RDG_LOG_DEBUG(...) IF_LOG_DEBUG("RenderGraph", __VA_ARGS__)
#define RDG_LOG_INFO(...) IF_LOG_INFO("RenderGraph", __VA_ARGS__)
#define RDG_LOG_WARNING(...) IF_LOG_WARNING("RenderGraph", __VA_ARGS__)
#define RDG_LOG_ERROR(...) IF_LOG_ERROR("RenderGraph", __VA_ARGS__)
#define RDG_LOG_CRITICAL(...) IF_LOG_CRITICAL("RenderGraph", __VA_ARGS__)

#define RDG_ASSERTION(expr, ...) IF_LOG_ASSERTION("RenderGraph", expr, __VA_ARGS__)
#define RDG_NOTNULL(ptr, ...) IF_LOG_ASSERTION("RenderGraph", (ptr) != nullptr, __VA_ARGS__)

    template <typename T, typename U> bool HasFlagBit(T value, U flag)
    {
        return (static_cast<std::underlying_type_t<T>>(value) & static_cast<std::underlying_type_t<U>>(flag)) != 0;
    }

#define RDG_NOT_IMPLEMENTED() RDG_LOG_CRITICAL("Function not implemented: {}", __FUNCTION__)

} // namespace Ifrit::Runtime::RenderCore::RDG