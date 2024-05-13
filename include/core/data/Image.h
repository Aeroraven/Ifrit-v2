#pragma once
#include "./core/definition/CoreExports.h"

namespace Ifrit::Core::Data {
	template<typename T>
	class Image {
	private:
		std::vector<T> data;
		size_t width;
		size_t height;
		size_t channel;

		size_t widthMulChannel;
	public:
		Image(size_t width, size_t height, size_t channel) : width(width), height(height), channel(channel) {
			data.resize(width * height * channel);
			widthMulChannel = width * channel;
		}
		Image(size_t width, size_t height, size_t channel, const std::vector<T>& data) : width(width), height(height), channel(channel), data(data) {
			ifritAssert(data.size() == width * height * channel, "Data size does not match the image size");
			widthMulChannel = width * channel;
		}
		Image(size_t width, size_t height, size_t channel, std::vector<T>&& data) : width(width), height(height), channel(channel), data(std::move(data)) {
			ifritAssert(data.size() == width * height * channel, "Data size does not match the image size");
			widthMulChannel = width * channel;
		}
		Image(const Image& other) : width(other.width), height(other.height), channel(other.channel), data(other.data) {
			widthMulChannel = width * channel;
		}
		Image(Image&& other) noexcept : width(other.width), height(other.height), channel(other.channel), data(std::move(other.data)) {
			widthMulChannel = width * channel;
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
			const auto p = y * widthMulChannel + x * channel;
			data[p + 0] = r;
			data[p + 1] = g;
			data[p + 2] = b;
			data[p + 3] = a;
		}

		void fillArea(size_t x, size_t y, size_t w, size_t h, const T& value) {
			ifritAssert(x + w <= width && y + h <= height, "Area out of range");
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
			data.clear();
			data.resize(width * height * channel);
			if(value != 0) std::fill(data.begin(), data.end(), value);
		}

		T& operator()(size_t x, size_t y, size_t c) {
			ifritAssert(x < width && y < height && c < channel, "Index out of range");
			return data[y * width * channel + x * channel + c];
		}

		const T& operator()(size_t x, size_t y, size_t c) const {
			ifritAssert(x < width && y < height && c < channel, "Index out of range");
			return data[y * width * channel + x * channel + c];
		}

		const T* getData() const {
			return data.data();
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