#pragma once
#include "RhiBaseTypes.h"
#include "ifrit/rhi/common/RhiApi.h"
#include "ifrit/rhi/common/RhiResource.h"
#include <any>

namespace Ifrit::RHI
{
    struct RhiShaderParameter;
    struct RhiShaderVariantDesc;
    struct RhiShaderCreateDesc;

    struct RhiShaderCreateDesc
    {
        String               mName;
        String               mFilePath;
        String               mEntryPoint;
        ERhiShaderStage      mStage      = ERhiShaderStage::Vertex;
        ERhiShaderSourceType mSourceType = ERhiShaderSourceType::SlangCode;
    };

    // ===== Shader Resource =====

    class IFRIT_RHI_API RhiShaderVariant
    {
    public:
        virtual ~RhiShaderVariant() = default;

        virtual u64          GetSignatureHash() const                                   = 0;
        virtual RhiRawHandle GetRawHandle() const                                       = 0;
        virtual bool         ValidateShaderParameters(const RhiShaderParameter& params) = 0;

        virtual u32          GetRefl_PushConstantSize() const = 0;
    };

    class IFRIT_RHI_API RhiShader : public RhiDeviceResource
    {
    public:
        RhiShader(RhiShaderCreateDesc desc) : RhiDeviceResource(ERhiResourceType::Shader), mDesc(desc) {}
        virtual ~RhiShader() = default;

        virtual RhiShaderVariant* GetVariant(const Vec<String>& keys) = 0;
        virtual bool              IsMultiCompileReady()               = 0;

    protected:
        RhiShaderCreateDesc mDesc;
    };

    struct RhiShaderVariantDesc
    {
        RhiShaderRef      mShader  = nullptr;
        RhiShaderVariant* mVariant = nullptr;

        bool              operator==(const RhiShaderVariantDesc& other) const
        {
            return (mShader == other.mShader) && (mVariant == other.mVariant);
        }

        u64 Hash() const
        {
            u64 addr1 = reinterpret_cast<u64>(mShader.get());
            u64 addr2 = reinterpret_cast<u64>(mVariant);
            return addr1 ^ addr2;
        }
    };

    // ===== Shader Registry =====
    struct RhiShaderRegistryInternal;
    class IFRIT_RHI_API RhiShaderRegistry
    {
    public:
        RhiShaderRegistry();
        virtual ~RhiShaderRegistry();
        virtual RhiShaderRef RegisterShader(const RhiShaderCreateDesc& desc);
        virtual RhiShaderRef GetShader(const String& name);

        virtual void         RequestShaderRegistryUnload();

    private:
        RhiShaderRegistryInternal* mInternal = nullptr;
    };

    // ===== Shader Parameters =====
    struct IFRIT_RHI_API RhiShaderParameter
    {
    public:
        template <typename T> bool SetValue(const String& name, const T& value)
        {
            mParameters[name] = value;
            return true;
        }

        template <typename T> T* GetValue(const String& name)
        {
            auto it = mParameters.find(name);
            if (it != mParameters.end())
            {
                return std::any_cast<T>(&it->second);
            }
            return nullptr;
        }

        const HashMap<String, std::any>& GetAllParameters() const { return mParameters; }

    protected:
        HashMap<String, std::any> mParameters;
    };

    // ===== Shader Global Functions =====
    IFRIT_RHI_API u32 GetMaxMultiCompileDirectives();

} // namespace Ifrit::RHI