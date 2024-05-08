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
	public:
		Image(size_t width, size_t height, size_t channel) : width(width), height(height), channel(channel) {
			data.resize(width * height * channel);
		}
		Image(size_t width, size_t height, size_t channel, const std::vector<T>& data) : width(width), height(height), channel(channel), data(data) {
			ifritAssert(data.size() == width * height * channel, "Data size does not match the image size");	
		}
		Image(size_t width, size_t height, size_t channel, std::vector<T>&& data) : width(width), height(height), channel(channel), data(std::move(data)) {
			ifritAssert(data.size() == width * height * channel, "Data size does not match the image size");
		}
		Image(const Image& other) : width(other.width), height(other.height), channel(other.channel), data(other.data) {}
		Image(Image&& other) noexcept : width(other.width), height(other.height), channel(other.channel), data(std::move(other.data)) {}

		T& operator()(size_t x, size_t y, size_t c) {
			ifritAssert(x < width && y < height && c < channel, "Index out of range");
			return data[y * width * channel + x * channel + c];
		}

		const T& operator()(size_t x, size_t y, size_t c) const {
			ifritAssert(x < width && y < height && c < channel, "Index out of range");
			return data[y * width * channel + x * channel + c];
		}

		T* getData() {
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