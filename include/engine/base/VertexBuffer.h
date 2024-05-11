#pragma once

#include "core/utility/CoreUtils.h"
#include "engine/base/BufferLayout.h"
#include "engine/base/TypeDescriptor.h"
namespace Ifrit::Engine {
	class VertexBuffer {
	private:
		std::vector<uint8_t> buffer;
		std::vector<TypeDescriptor> layout;
		std::vector<int> offsets;
		int vertexCount;
		int elementSize;
	public:
		void setLayout(const std::vector<TypeDescriptor>& layout);

		void allocateBuffer(const size_t numVertices) { 
			int elementSize = 0;
			for (int i = 0; i < layout.size(); i++) {
				elementSize += layout[i].size;
			}
			buffer.resize(numVertices * elementSize);
			this->vertexCount = numVertices;
		}

		template<class T>
		inline T getValue(const int index, const int attribute) const{
			size_t dOffset = offsets[attribute] + index * elementSize;
			const char* data = reinterpret_cast<const char*>(&buffer[dOffset]);
			return *reinterpret_cast<const T*>(data);
		}

		template<class T>
		inline const T* getValuePtr(const int index, const int attribute) const {
			size_t dOffset = offsets[attribute] + index * elementSize;
			const char* data = reinterpret_cast<const char*>(&buffer[dOffset]);
			return reinterpret_cast<const T*>(data);
		}

		template<class T>
		inline T setValue(const int index, const int attribute, const T value) {
			size_t dOffset = offsets[attribute] + index * elementSize;
			char* data = reinterpret_cast<char*>(&buffer[dOffset]);
			*reinterpret_cast<T*>(data) = value;
			return value;
		}

		void setVertexCount(const int vertexCount);
		int getVertexCount() const;
		int getAttributeCount() const;
		TypeDescriptor getAttributeDescriptor(int index) const;

	};
}