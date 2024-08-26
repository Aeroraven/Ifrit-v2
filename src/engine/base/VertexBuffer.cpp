#include "engine/base/VertexBuffer.h"
#include "engine/base/TypeDescriptor.h"

namespace Ifrit::Engine {

	IFRIT_APIDECL void VertexBuffer::setLayout(const std::vector<TypeDescriptor>& layout) {
		this->layout = layout;
		this->offsets.resize(layout.size());
		int offset = 0;
		for (int i = 0; i < layout.size(); i++) {
			offsets[i] = offset;
			offset += layout[i].size;
			if (layout[i].type == TypeDescriptorEnum::IFTP_UNDEFINED) {
				printf("Undefined layout %d\n",TypeDescriptors.FLOAT4.type);
				std::abort();
			}
		}
		elementSize = offset;
	}

	IFRIT_APIDECL void VertexBuffer::setVertexCount(const int vertexCount){
		this->vertexCount = vertexCount;
	}

	IFRIT_APIDECL int VertexBuffer::getVertexCount() const{
		return vertexCount;
	}

	IFRIT_APIDECL int VertexBuffer::getAttributeCount() const{
		return layout.size();
	}

	IFRIT_APIDECL TypeDescriptor VertexBuffer::getAttributeDescriptor(int index) const{
		return layout[index];
	}
	
}