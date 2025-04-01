
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "ifrit/core/base/IfritBase.h"
#include "ifrit/softgraphics/core/definition/CoreExports.h"

namespace Ifrit::Graphics::SoftGraphics::Core::Data
{
    template <typename T> class IFRIT_APIDECL Image
    {
    private:
        T*           data = nullptr;

        bool         isCudaPinned = false;
        const size_t width;
        const size_t height;
        const size_t channel;

    public:
        Image() : width(0), height(0), channel(0) {}
        Image(size_t width, size_t height, size_t channel, bool pinned)
            : width(width), height(height), channel(channel), isCudaPinned(pinned)
        {
#ifdef IFRIT_FEATURE_CUDA
            if (isCudaPinned)
            {
                cudaMallocHost(&data, width * height * channel * sizeof(T));
            }
            else
            {
                data = new T[width * height * channel];
            }
#else
            data = new T[width * height * channel];
#endif
        }
        Image(size_t width, size_t height, size_t channel) : width(width), height(height), channel(channel)
        {
            data = new T[width * height * channel];
        }
        Image(const Image& other) : width(other.width), height(other.height), channel(other.channel)
        {
            data = new T[width * height * channel];
            memcpy(data, other.data, width * height * channel * sizeof(T));
        }
        Image(Image&& other) noexcept : width(other.width), height(other.height), channel(other.channel)
        {
            data       = other.data;
            other.data = nullptr;
        }

        ~Image()
        {
            if (data != nullptr)
            {
#ifdef IFRIT_FEATURE_CUDA
                if (isCudaPinned)
                {
                    cudaFreeHost(data);
                }
                else
                {
                    delete[] data;
                }
#else
                delete[] data;
#endif
            }
        }

        void fillAreaRGBA(size_t x, size_t y, size_t w, size_t h, const T& r, const T& g, const T& b, const T& a)
        {
            for (size_t i = y; i < y + h; i++)
            {
                for (size_t j = x; j < x + w; j++)
                {
                    data[i * width * channel + j * channel + 0] = r;
                    data[i * width * channel + j * channel + 1] = g;
                    data[i * width * channel + j * channel + 2] = b;
                    data[i * width * channel + j * channel + 3] = a;
                }
            }
        }

        inline void fillPixelRGBA(size_t x, size_t y, const T& r, const T& g, const T& b, const T& a)
        {
            // ifritAssert(x < width && y < height, "Pixel out of range");
            data[y * width * channel + x * channel + 0] = r;
            data[y * width * channel + x * channel + 1] = g;
            data[y * width * channel + x * channel + 2] = b;
            data[y * width * channel + x * channel + 3] = a;
        }

        inline void fillPixelRGBA128ps(size_t x, size_t y, const __m128& ps)
        {
            _mm_store_ps(&data[y * width * channel + x * channel], ps);
        }

        inline T* getPixelRGBAUnsafe(size_t x, size_t y) { return &data[y * width * channel + x * channel]; }

        void      fillArea(size_t x, size_t y, size_t w, size_t h, const T& value)
        {
            // ifritAssert(x + w <= width && y + h <= height, "Area out of range");
            for (size_t i = y; i < y + h; i++)
            {
                for (size_t j = x; j < x + w; j++)
                {
                    for (size_t c = 0; c < channel; c++)
                    {
                        data[i * width * channel + j * channel + c] = value;
                    }
                }
            }
        }

        void fillAreaF(float ndcX, float ndcY, float ndcW, float ndcH, const T& value)
        {
            size_t x = ndcX * width;
            size_t y = ndcY * height;
            size_t w = ndcW * width;
            size_t h = ndcH * height;
            fillArea(x, y, w, h, value);
        }

        T*   getData() { return data; }

        void clearImage(T value = 0)
        {
            const auto dataPtr  = data;
            const auto dataSize = width * height * channel;
            if IF_CONSTEXPR (std::is_same_v<T, char> && !std::is_same_v<T, unsigned char>)
            {
                memset(dataPtr, value, dataSize * sizeof(T));
            }
            else
            {
                if (value != 0)
                    IFRIT_BRANCH_UNLIKELY
                    {
                        auto st = data;
                        auto ed = data + width * height * channel;
                        while (st != ed)
                        {
                            *st = value;
                            st++;
                        }
                    }
                else
                {
                    memset(dataPtr, 0, dataSize);
                }
            }
        }

        void clearImageMultithread(T value, int workerId, int totalWorkers)
        {
            const auto dataPtr    = data;
            const auto dataSizeSt = width * height * channel * workerId / totalWorkers;
            const auto dataSizeEd = width * height * channel * (workerId + 1) / totalWorkers;

            if IF_CONSTEXPR (std::is_same_v<T, char> && !std::is_same_v<T, unsigned char>)
            {
                memset(dataPtr + dataSizeSt, value, (dataSizeEd - dataSizeSt) * sizeof(T));
            }
            else
            {
                auto st = data + dataSizeSt;
                auto ed = data + dataSizeEd;
#ifdef _MSC_VER
                // Check if satisfies 0x(pp)(pp)(pp)(pp)
                if IF_CONSTEXPR (sizeof(T) == 4)
                {
                    auto hex = std::bit_cast<unsigned, T>(value);
                    if (hex == (hex & 0xFF) * 0x01010101)
                    {
                        memset(st, (hex & 0xFF), (ed - st) * sizeof(T));
                    }
                    else
                    {
                        std::fill(st, ed, value);
                    }
                }
                else
                {
                    std::fill(st, ed, value);
                }
#else
                std::fill(st, ed, value);
#endif
            }
        }

        void clearImageZero()
        {
            const auto dataPtr  = data;
            const auto dataSize = width * height * channel;
            memset(dataPtr, 0, dataSize * sizeof(T));
        }

        void clearImageZeroMultiThread(int workerId, int totalWorkers)
        {
            const auto dataPtr    = data;
            const auto dataSizeSt = width * height * channel * workerId / totalWorkers;
            const auto dataSizeEd = width * height * channel * (workerId + 1) / totalWorkers;
            memset(dataPtr + dataSizeSt, 0, (dataSizeEd - dataSizeSt) * sizeof(T));
        }

        inline T& operator()(size_t x, size_t y, size_t c) { return data[y * width * channel + x * channel + c]; }

        const T&  operator()(size_t x, size_t y, size_t c) const { return data[y * width * channel + x * channel + c]; }

        const T*  getData() const { return data; }

        IF_CONSTEXPR size_t GetSizeOf() const { return sizeof(T); }

        IF_CONSTEXPR size_t GetWidth() const { return width; }

        IF_CONSTEXPR size_t GetHeight() const { return height; }

        IF_CONSTEXPR size_t getChannel() const { return channel; }

        IF_CONSTEXPR size_t GetSize() const { return width * height * channel; }
    };

    using ImageF32 = Image<float>;
    using ImageU8  = Image<uint8_t>;
    using ImageU16 = Image<uint16_t>;
    using ImageU32 = Image<u32>;
    using ImageU64 = Image<u64>;
    using ImageI8  = Image<int8_t>;
    using ImageI16 = Image<int16_t>;
    using ImageI32 = Image<int32_t>;
    using ImageI64 = Image<int64_t>;
} // namespace Ifrit::Graphics::SoftGraphics::Core::Data