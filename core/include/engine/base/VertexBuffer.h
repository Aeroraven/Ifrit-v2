#pragma once

#include "core/utility/CoreUtils.h"
#include "engine/base/TypeDescriptor.h"
namespace Ifrit::Engine {

	struct VertexBufferContext {
		std::vector<uint8_t> buffer;
		std::vector<TypeDescriptor> layout;
		std::vector<int> offsets;
	};

	class IFRIT_APIDECL VertexBuffer {
	private:
		VertexBufferContext* context;
		int vertexCount;
		int elementSize;
	public:
		VertexBuffer();
		~VertexBuffer();
		void setLayout(const std::vector<TypeDescriptor>& layout);
		void allocateBuffer(const size_t numVertices);
		void setVertexCount(const int vertexCount);
		int getVertexCount() const;
		int getAttributeCount() const;
		TypeDescriptor getAttributeDescriptor(int index) const;

		inline int getOffset(int i) const {
			return context->offsets[i];
		}
		inline int getElementSize() const {
			return elementSize;
		}

		/* Templates */
		template<class T>
		inline T getValue(const int index, const int attribute) const{
			size_t dOffset = context->offsets[attribute] + index * elementSize;
			const char* data = reinterpret_cast<const char*>(&context->buffer[dOffset]);
			return *reinterpret_cast<const T*>(data);
		}

		template<class T>
		inline const T* getValuePtr(const int index, const int attribute) const {
			size_t dOffset = context->offsets[attribute] + index * elementSize;
			const char* data = reinterpret_cast<const char*>(&context->buffer[dOffset]);
			return reinterpret_cast<const T*>(data);
		}

		template<class T>
		inline T setValue(const int index, const int attribute, const T value) {
			size_t dOffset = context->offsets[attribute] + index * elementSize;
			char* data = reinterpret_cast<char*>(&context->buffer[dOffset]);
			*reinterpret_cast<T*>(data) = value;
			return value;
		}

		/* Inline */
		inline char* getBufferUnsafe() const {
			return (char*)context->buffer.data();
		}

		inline uint32_t getBufferSize() const {
			return context->buffer.size();
		}

		/* DLL Compatible */
		void setLayoutCompatible(const TypeDescriptor* layouts, int num);
		void setValueFloat4Compatible(const int index, const int attribute, const ifloat4 value);
	};
}