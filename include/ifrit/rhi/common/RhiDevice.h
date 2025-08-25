#pragma once
#include "RhiApi.h"
#include "RhiBaseTypes.h"

namespace Ifrit::RHI
{
    struct RhiCommandListExecutor;

    struct RhiCapabilityList
    {
        // Rhi capabilities
        bool bValidationLayerEnabled = true;
        bool bImmediateMode          = false;
        bool bAsyncComputeEnable     = true;

        // Device capabilities
        bool bMeshShaderEnabled                = true;
        bool bHardwareRayTracingEnabled        = false;
        bool bConservativeRasterizationEnabled = true;
        bool bShaderFloatAtomicsEnabled        = true;

        // Experimental options
        bool bTreatConstantBufferAsStorageBuffer = true;
    };

    struct RhiPropertyList
    {
        u32 mWaveSize = ~0u;
    };

    struct RhiDeviceProcs;

    // ===== RhiDevice Interface =====

    // UPD 250325: Resource removal algo before destroys the resource that still in use on device side
    // referencing Unreal's resource state management, a delete queue should be maintained

    class IFRIT_RHI_API IRhiDeviceResourceDeleteQueue
    {
    public:
        virtual void AddResourceToDeleteQueue(RhiDeviceResource* resource) = 0;
        virtual i32  ProcessDeleteQueue()                                  = 0;
    };

    class IFRIT_APIDECL RhiDevice
    {
    public:
        virtual RhiCapabilityList              GetCapabilities() const        = 0;
        virtual RhiPropertyList                GetProperties() const          = 0;
        virtual IRhiDeviceResourceDeleteQueue* GetResourceDeleteQueue()       = 0;
        virtual RhiDeviceProcs*                GetDeviceProcs() const         = 0;
        virtual RhiCommandListExecutor*        GetCommandListExecutor() const = 0;
    };

    class IFRIT_APIDECL RhiDeviceChild
    {
    protected:
        RhiDevice* mContext;

    public:
        inline RhiDevice* GetDevice() const { return mContext; }
    };

    // ===== Swapchain =====
    class IFRIT_APIDECL RhiSwapchain : public RhiDeviceChild
    {
    public:
        virtual ~RhiSwapchain()                   = default;
        virtual void Present()                    = 0;
        virtual u32  AcquireNextImage()           = 0;
        virtual u32  GetNumBackbuffers() const    = 0;
        virtual u32  GetCurrentFrameIndex() const = 0;
        virtual u32  GetCurrentImageIndex() const = 0;
    };

} // namespace Ifrit::RHI