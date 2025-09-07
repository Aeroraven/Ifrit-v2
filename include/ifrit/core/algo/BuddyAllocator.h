#pragma once
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/Intrinsics.h"
namespace Ifrit
{

    IFRIT_CORE_API void BuddyAllocatorReportCritical(const String& msg);

    struct BuddyBlock
    {
        u64 mOffset;
        u64 mOrder;
    };

    struct BuddyAllocation
    {
        bool mSuccess = false;
        u64  mOffset  = 0;
    };

    class BuddyAddressAllocator
    {
    public:
        BuddyAddressAllocator(u64 granularity, u64 totalSize)
        {
            mTotalMinBlocks = totalSize / granularity;
            mGranularity    = granularity;
            mTotalSize      = totalSize;
            mOrders         = Math::IntegerLog2(mTotalMinBlocks) + 1;

            mFreeBlocks.resize(mOrders);
            if ((mTotalMinBlocks & (mTotalMinBlocks - 1)) != 0)
            {
                BuddyAllocatorReportCritical("TotalSize must be power of two multiple of granularity");
            }
            mFreeBlocks[mOrders - 1].insert(0);
        }

        BuddyAllocation Allocate(u64 size)
        {
            u64             requiredBlocks = (size + mGranularity - 1) / mGranularity;
            BuddyAllocation ret;
            ret.mOffset = AllocateInternal(requiredBlocks);
            if (ret.mOffset != ~0ull)
            {
                ret.mSuccess                  = true;
                mAllocatedBlocks[ret.mOffset] = { ret.mOffset / mGranularity,
                    static_cast<u64>(Math::IntegerLog2RoundUp(requiredBlocks)) };
            }
            return ret;
        }

        void Free(u64 offset)
        {
            auto it = mAllocatedBlocks.find(offset);
            if (it == mAllocatedBlocks.end())
            {
                BuddyAllocatorReportCritical("Freeing invalid or already freed block");
                return;
            }
            ReturnBlock(it->second.mOffset, it->second.mOrder);
            mAllocatedBlocks.erase(it);
        }

        void DumpFreeBlocks()
        {
            printf("BuddyAllocator DumpFreeBlocks:\n");
            for (u64 order = 0; order < mOrders; order++)
            {
                printf(" Order %llu: ", order);
                for (auto offset : mFreeBlocks[order])
                {
                    printf("%llu ", offset);
                }
                printf("\n");
            }
        }

    protected:
        u64 RequestBlock(u64 requiredOrder)
        {
            if (requiredOrder >= mOrders)
            {
                return ~0ull;
            }
            if (mFreeBlocks[requiredOrder].size() > 0)
            {
                auto it     = mFreeBlocks[requiredOrder].begin();
                u64  offset = *it;
                mFreeBlocks[requiredOrder].erase(it);
                return offset;
            }
            else
            {
                u64 higherBlock = RequestBlock(requiredOrder + 1);
                if (higherBlock == ~0ull)
                {
                    return ~0ull;
                }
                u64 buddy = higherBlock + (1ull << requiredOrder);
                mFreeBlocks[requiredOrder].insert(buddy);
                return higherBlock;
            }
        }

        void ReturnBlock(u64 offset, u64 order)
        {
            if (order >= mOrders)
            {
                BuddyAllocatorReportCritical("ReturnBlock: order out of range");
                return;
            }
            u64  buddy = offset ^ (1ull << order);
            auto it    = mFreeBlocks[order].find(buddy);
            if (it != mFreeBlocks[order].end())
            {
                mFreeBlocks[order].erase(it);
                ReturnBlock(std::min(offset, buddy), order + 1);
            }
            else
            {
                mFreeBlocks[order].insert(offset);
            }
        }

        u64 AllocateInternal(u64 requiredBlocks)
        {
            u64 requiredOrder = Math::IntegerLog2RoundUp(requiredBlocks);
            if (requiredOrder >= mOrders)
            {
                return ~0ull;
            }
            auto blockOffset = RequestBlock(requiredOrder);
            if (blockOffset == ~0ull)
            {
                return ~0ull;
            }
            return blockOffset * mGranularity;
        }

    private:
        u64                      mGranularity;
        u64                      mTotalSize;
        u64                      mOrders;

        u64                      mTotalMinBlocks;

        Vec<Set<u64>>            mFreeBlocks; // key is offset
        HashMap<u64, BuddyBlock> mAllocatedBlocks;
    };

} // namespace Ifrit