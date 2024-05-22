#pragma once
#include "./core/definition/CoreExports.h"

namespace Ifrit::Core::Data {
	template<typename T>
	class Image {
	private:
		T* data = nullptr;
		
		bool isCudaPinned = false;
		const size_t width;
		const size_t height;
		const size_t channel;
	public:
		Image() : width(0), height(0), channel(0) {}
		Image(size_t width, size_t height, size_t channel, bool pinned) : width(width), height(height), channel(channel), isCudaPinned(pinned) {
			if (isCudaPinned) {
				cudaMallocHost(&data, width * height * channel * sizeof(T));
			}
			else {
				data = new T[width * height * channel];
			}
		}
		Image(size_t width, size_t height, size_t channel) : width(width), height(height), channel(channel) {
			data = new T[width * height * channel];
		}
		Image(const Image& other) : width(other.width), height(other.height), channel(other.channel) {
			data = new T[width * height * channel];
			memcpy(data, other.data, width * height * channel * sizeof(T));
		}
		Image(Image&& other) noexcept : width(other.width), height(other.height), channel(other.channel){
			data = other.data;
			other.data = nullptr;
		}

	
		~Image() {
			if (data != nullptr) {
				if (isCudaPinned) {
					cudaFreeHost(data);
				}
				else {
					delete[] data;
				}
			}
		}

		void fillAreaRGBA(size_t x, size_t y, size_t w, size_t h, const T& r, const T& g, const T& b, const T& a) {
			for (size_t i = y; i < y + h; i++) {
				for (size_t j = x; j < x + w; j++) {
					data[i * width * channel + j * channel + 0] = r;
					data[i * width * channel + j * channel + 1] = g;
					data[i * width * channel + j * channel + 2] = b;
					data[i * width * channel + j * channel + 3] = a;
				}
			}
		}

		inline void fillPixelRGBA(size_t x, size_t y, const T& r, const T& g, const T& b, const T& a) {
			//ifritAssert(x < width && y < height, "Pixel out of range");
			data[y * width * channel + x * channel + 0] = r;
			data[y * width * channel + x * channel + 1] = g;
			data[y * width * channel + x * channel + 2] = b;
			data[y * width * channel + x * channel + 3] = a;
		}

		inline void fillPixelRGBA128ps (size_t x, size_t y, const __m128& ps) {
			_mm_store_ps(&data[y * width * channel + x * channel], ps);
		}

		void fillArea(size_t x, size_t y, size_t w, size_t h, const T& value) {
			//ifritAssert(x + w <= width && y + h <= height, "Area out of range");
			for (size_t i = y; i < y + h; i++) {
				for (size_t j = x; j < x + w; j++) {
					for (size_t c = 0; c < channel; c++) {
						data[i * width * channel + j * channel + c] = value;
					}
				}
			}
		}

		void fillAreaF(float ndcX, float ndcY, float ndcW, float ndcH, const T& value) {
			size_t x = ndcX * width;
			size_t y = ndcY * height;
			size_t w = ndcW * width;
			size_t h = ndcH * height;
			fillArea(x, y, w, h, value);
		}

		void clearImage(T value=0) {
			const auto dataPtr = data;
			const auto dataSize = width * height * channel;
			if constexpr (std::is_same_v<T, char> && !std::is_same_v<T, unsigned char>) {
				memset(dataPtr, value, dataSize * sizeof(T));
			}
			else {
				if (value != 0) IFRIT_BRANCH_UNLIKELY{
					auto st = data;
					auto ed = data + width*height*channel;
					while (st != ed) {
						*st = value;
						st++;
					}
				}
				else {
					memset(dataPtr, value, dataSize);
				}
			}
		}

		void clearImageZero() {
			const auto dataPtr = data;
			const auto dataSize = width * height * channel;
			memset(dataPtr, 0, dataSize * sizeof(T));
		}

		T& operator()(size_t x, size_t y, size_t c) {
			return data[y * width * channel + x * channel + c];
		}

		const T& operator()(size_t x, size_t y, size_t c) const {
			return data[y * width * channel + x * channel + c];
		}

		const T* getData() const {
			return data;
		}

		constexpr size_t getSizeOf() const {
			return sizeof(T);
		}

		constexpr size_t getWidth() const {
			return width;
		}

		constexpr size_t getHeight() const {
			return height;
		}

		constexpr size_t getChannel() const {
			return channel;
		}

		constexpr size_t getSize() const {
			return width * height * channel;
		}

	};

	using ImageF32 = Image<float>;
	using ImageU8 = Image<uint8_t>;
	using ImageU16 = Image<uint16_t>;
	using ImageU32 = Image<uint32_t>;
	using ImageU64 = Image<uint64_t>;
	using ImageI8 = Image<int8_t>;
	using ImageI16 = Image<int16_t>;
	using ImageI32 = Image<int32_t>;
	using ImageI64 = Image<int64_t>;
}