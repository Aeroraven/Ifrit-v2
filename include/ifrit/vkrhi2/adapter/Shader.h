#pragma once
#include "ifrit/vkrhi2/common/VkAdapterApi.h"
#include "ifrit/vkrhi2/common/Pch.h"
#include "ifrit/rhi/common/RhiShaderResource.h"

#include <vulkan/vulkan.h>

namespace Ifrit::RHI::VulkanRHI2
{
    class VA_Device;

    struct VA_ShaderVariantCI
    {
        String               mIRCode;
        RHI::ERhiShaderStage mStage;
        String               mShaderName;
        String               mEntryPoint;
    };

    struct VA_ShaderVariantInternal;
    class IFRIT_VKRHI2_API VA_ShaderVariant : public RhiShaderVariant
    {
    public:
        VA_ShaderVariant(VA_Device* device, const VA_ShaderVariantCI& ci, void* reflData);
        virtual ~VA_ShaderVariant();

        virtual u64                             GetSignatureHash() const override;
        virtual RhiRawHandle                    GetRawHandle() const override;
        virtual bool                            ValidateShaderParameters(const RhiShaderParameter& params) override;

        virtual u32                             GetRefl_PushConstantSize() const override;

    public:
        virtual VkPipelineShaderStageCreateInfo GetShaderStageInfo() const;

    private:
        VA_ShaderVariantInternal* mData;
    };

    struct VA_ShaderInternal;
    class IFRIT_VKRHI2_API VA_Shader : public RhiShader
    {
    public:
        VA_Shader(VA_Device* device, const RhiShaderCreateDesc& desc);
        virtual ~VA_Shader();

        virtual RhiShaderVariant* GetVariant(const Vec<String>& keys) override;
        virtual bool              IsMultiCompileReady() override;

    private:
        void CompileShaderVariant(u64 permId);
        void PrecompileMultiCompileShaders();
        void PrecompileMultiCompileShadersImpl(u32 curVariantTag, u64 curPermId);

    private:
        VA_ShaderInternal* mData;
    };

} // namespace Ifrit::RHI::VulkanRHI2