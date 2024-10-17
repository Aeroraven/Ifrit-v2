#pragma once
#include <stdexcept>
namespace Ifrit::Engine::VkRenderer{
    inline void vkrAssert(bool condition, const char* message){
        if(!condition){
            throw std::runtime_error(message);
        }
    }
    inline void vkrDebug(const char* message){
        printf("%s\n", message);
    }
    inline void vkrVulkanAssert(VkResult result, const char* message){
        if(result != VK_SUCCESS){
            printf("Error code: %d\n", result);
            throw std::runtime_error(message);
        }
    }
    inline void vkrLog(const char* message){
        printf("%s\n", message);
    }
    inline void vkrError(const char* message){
        fprintf(stderr, "%s\n", message);
        throw std::runtime_error(message);
    }
}