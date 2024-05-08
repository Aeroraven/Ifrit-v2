#pragma once

#include "core/utility/CoreUtils.h"
#include "engine/base/BufferLayout.h"

namespace Ifrit::Engine {
	class VertexBuffer {
	private:
		std::vector<uint8_t> buffer;
		std::vector<BufferLayout> layout;
		std::vector<int> offsets;
	public:
		void setLayout(const std::vector<BufferLayout>& layout);

		template<class T>
		inline void getValue(const int index, const int attribute) {
			size_t dOffset = offsets[attribute] + index * layout[attribute].size;
			char* data = reinterpret_cast<char*>(&buffer[dOffset]);
			return *reinterpret_cast<T*>(data);
		}
	};
}