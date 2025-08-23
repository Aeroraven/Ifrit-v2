#include "ifrit/rhi/common/RhiResource.h"
#include "ifrit/core/logging/Logging.h"
namespace Ifrit::RHI
{
    inline RhiTexture* RhiResourceView::GetUnderlyingTexture() const
    {
        IF_LOG_ASSERTION("RhiResourceView", mTexture != nullptr, "Underlying texture is null");
        return mTexture;
    }

    inline RhiBuffer* RhiResourceView::GetUnderlyingBuffer() const
    {
        IF_LOG_ASSERTION("RhiResourceView", mBuffer != nullptr, "Underlying buffer is null");
        return mBuffer;
    }

    void RhiResourceView::InternalCheck()
    {
        bool hasTexture    = mTexture != nullptr;
        bool hasBuffer     = mBuffer != nullptr;
        bool isTextureView = mDesc.mType == ERhiResourceViewedType::Texture;
        bool isBufferView  = mDesc.mType == ERhiResourceViewedType::Buffer;

        IF_LOG_ASSERTION("RhiResourceView", !(hasTexture && hasBuffer), "Has both texture and buffer");
        IF_LOG_ASSERTION("RhiResourceView", (hasTexture && isTextureView) || (hasBuffer && isBufferView),
            "Resource view type does not match underlying resource type");
    }

} // namespace Ifrit::RHI