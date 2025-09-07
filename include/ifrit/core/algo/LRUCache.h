#pragma once
#include "ifrit/core/base/CoreBase.h"
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/core/math/Intrinsics.h"
#include <list>

namespace Ifrit
{
    IFRIT_CORE_API void LRUCacheReportCritical(const String& msg);

    template <typename Key, typename Value> class LRUCache
    {
    private:
        struct CacheItem
        {
            Key   key;
            Value value;

            CacheItem(const Key& k, const Value& v) : key(k), value(v) {}
        };

        using CacheList     = std::list<CacheItem>;
        using CacheIterator = typename CacheList::iterator;
        using CacheMap      = HashMap<Key, CacheIterator>;

        mutable CacheList mItems;    // Doubly linked list for LRU ordering
        mutable CacheMap  mCacheMap; // Hash map for O(1) lookup
        u64               mCapacity; // Maximum number of items
        mutable u64       mSize;     // Current number of items

    public:
        explicit LRUCache(u64 capacity) : mCapacity(capacity), mSize(0)
        {
            if (capacity == 0)
            {
                LRUCacheReportCritical("LRU Cache capacity cannot be zero");
            }
        }

        ~LRUCache() = default;

        // Non-copyable for now (can be implemented if needed)
        LRUCache(const LRUCache&)            = delete;
        LRUCache& operator=(const LRUCache&) = delete;

        // Movable
        LRUCache(LRUCache&&)            = default;
        LRUCache& operator=(LRUCache&&) = default;

        // Get value by key, returns nullptr if not found
        // Moves accessed item to front (most recently used)
        Value*    Get(const Key& key) const
        {
            auto it = mCacheMap.find(key);
            if (it == mCacheMap.end())
            {
                return nullptr; // Not found
            }

            // Move to front (most recently used)
            auto listIt = it->second;
            mItems.splice(mItems.begin(), mItems, listIt);

            return &(listIt->value);
        }

        // Put key-value pair into cache
        // If key exists, updates value and moves to front
        // If cache is full, evicts least recently used item
        void Put(const Key& key, const Value& value)
        {
            auto it = mCacheMap.find(key);

            if (it != mCacheMap.end())
            {
                // Key exists, update value and move to front
                auto listIt   = it->second;
                listIt->value = value;
                mItems.splice(mItems.begin(), mItems, listIt);
                return;
            }

            // Key doesn't exist, add new item
            if (mSize >= mCapacity)
            {
                // Cache is full, evict least recently used (back of list)
                auto lastItem = mItems.back();
                mCacheMap.erase(lastItem.key);
                mItems.pop_back();
                mSize--;
            }

            // Add new item to front
            mItems.emplace_front(key, value);
            mCacheMap[key] = mItems.begin();
            mSize++;
        }

        // Check if key exists in cache
        bool Contains(const Key& key) const { return mCacheMap.find(key) != mCacheMap.end(); }

        // Remove item from cache
        bool Remove(const Key& key)
        {
            auto it = mCacheMap.find(key);
            if (it == mCacheMap.end())
            {
                return false; // Not found
            }

            auto listIt = it->second;
            mItems.erase(listIt);
            mCacheMap.erase(it);
            mSize--;
            return true;
        }

        // Clear all items
        void Clear()
        {
            mItems.clear();
            mCacheMap.clear();
            mSize = 0;
        }

        // Get current size
        u64  Size() const { return mSize; }

        // Get capacity
        u64  Capacity() const { return mCapacity; }

        // Check if cache is empty
        bool IsEmpty() const { return mSize == 0; }

        // Check if cache is full
        bool IsFull() const { return mSize >= mCapacity; }

        // Resize cache capacity
        void Resize(u64 newCapacity)
        {
            if (newCapacity == 0)
            {
                LRUCacheReportCritical("LRU Cache capacity cannot be zero");
                return;
            }

            mCapacity = newCapacity;

            // Evict items if new capacity is smaller
            while (mSize > mCapacity)
            {
                auto lastItem = mItems.back();
                mCacheMap.erase(lastItem.key);
                mItems.pop_back();
                mSize--;
            }
        }

        // Get all keys in LRU order (most recent first)
        Vec<Key> GetKeys() const
        {
            Vec<Key> keys;
            keys.reserve(mSize);

            for (const auto& item : mItems)
            {
                keys.push_back(item.key);
            }

            return keys;
        }

        // Debug: Get usage statistics
        struct CacheStats
        {
            u64 size;
            u64 capacity;
            f64 loadFactor;
        };

        CacheStats GetStats() const
        {
            return CacheStats{ mSize, mCapacity, static_cast<f64>(mSize) / static_cast<f64>(mCapacity) };
        }
    };

} // namespace Ifrit