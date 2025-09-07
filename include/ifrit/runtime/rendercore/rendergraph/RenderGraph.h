#pragma once
#include "ifrit/runtime/base/Base.h"
#include "ifrit/runtime/common/Pch.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include "ifrit/runtime/rendercore/rendergraph/RenderGraphResource.h"

#ifdef __INTELLISENSE__
    #define IFRIT_RDG_API
#else
    #define IFRIT_RDG_API IFRIT_RUNTIME_API
#endif

namespace Ifrit::Runtime::RDG
{
    class RDGResource;
    class RDGPass;
    class RDGGraphBuilder;
    class RDGTextureResource;
    class RDGBufferResource;
    class RDGScopeMarker;

    namespace ERDGResourceFlag
    {
        enum Enum
        {
            Imported  = 0x1,
            Uploading = 0x2,
        };
    } // namespace ERDGResourceFlag

    enum class ERDGResourceAccessFlag
    {
        SRVRead         = 0x1,
        UAVWrite        = 0x2,
        UAVRead         = 0x4,
        IndirectArgRead = 0x8,
        CopyDst         = 0x10,
        CopySrc         = 0x20,
        RenderTarget    = 0x40,
    };

    enum ERDGPassType
    {
        Invalid  = 0,
        Compute  = 1,
        Graphics = 2,
        Transfer = 3,
    };

    using ERDGResourceFlags = u32;

    // Resource Descriptions
    struct IRDGResourceViewRegistry
    {
        virtual RHI::RhiSRVDesc GetSRV(RDGResource* res) = 0;
        virtual RHI::RhiCBVDesc GetCBV(RDGResource* res);
        virtual RHI::RhiSRVDesc GetUAV(RDGResource* res, u32 mipLevel = 0, u32 arraySlice = 0);

        virtual void            DeclareRenderTarget(RDGTextureResource* res,
                       RHI::RhiRenderTargetLoadOp                       loadOp     = RHI::RhiRenderTargetLoadOp::Load,
                       RHI::RhiClearValue2                              clearValue = RHI::RhiClearValue2());
        virtual void            DeclareIndirectArgBuffer(RDGBufferResource* res, u32 offset = 0);
    };

    // Resources
    struct RDGResourceProperty;
    class IFRIT_RDG_API RDGResource
    {
    public:
        virtual ~RDGResource() = default;

    protected:
        RDGResource(const String& name, ERDGResourceFlags flags = 0);

    private:
        RDGResourceProperty* mProps;

        friend class RDGGraphBuilder;
    };

    struct RDGTextureResourceProperty;
    class IFRIT_RDG_API RDGTextureResource : public RDGResource
    {
    public:
        RHI::RhiTexture* GetRHI() const;

    protected:
        RDGTextureResource(RHI::RhiTexture* imported);
        RDGTextureResource(const RDGBufferDesc& desc, const String& name);

    private:
        RDGTextureResourceProperty* mProps;

        friend class RDGGraphBuilder;
    };

    struct RDGBufferResourceProperty;
    class IFRIT_RDG_API RDGBufferResource : public RDGResource
    {
    public:
        RHI::RhiBuffer* GetRHI() const;

    protected:
        RDGBufferResource(RHI::RhiBuffer* imported);
        RDGBufferResource(const RDGBufferDesc& desc, const String& name);

    private:
        RDGBufferResourceProperty* mProps;

        friend class RDGGraphBuilder;
    };

    // Passes
    struct RDGPassProperty;
    class IFRIT_RDG_API RDGPass
    {
    public:
        virtual ~RDGPass() = default;

    protected:
        RDGPass(const String& name, ERDGPassType type);

        void AddResourceDependency(const RDGResource& res, ERDGResourceAccessFlag accessFlags);

    private:
        RDGPassProperty* mProps;

        friend class RDGGraphBuilder;
    };

    // RDG Builder
    struct RDGGraphBuilderProperty;
    class IFRIT_RDG_API RDGGraphBuilder
    {
    public:
        RDGGraphBuilder();

    private:
        RDGGraphBuilderProperty* mProps;
    };
} // namespace Ifrit::Runtime::RDG