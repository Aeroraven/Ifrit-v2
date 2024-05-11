#include "engine/base/VertexBuffer.h"
#include "engine/base/TypeDescriptor.h"

namespace Ifrit::Engine {
	void VertexBuffer::setLayout(const std::vector<TypeDescriptor>& layout) {
		this->layout = layout;
		this->offsets.resize(layout.size());
		int offset = 0;
		for (int i = 0; i < layout.size(); i++) {
			offsets[i] = offset;
			offset += layout[i].size;
		}
		elementSize = offset;
	}

	void VertexBuffer::setVertexCount(const int vertexCount){
		this->vertexCount = vertexCount;
	}

	int VertexBuffer::getVertexCount() const{
		return vertexCount;
	}

	int VertexBuffer::getAttributeCount() const{
		return layout.size();
	}

	TypeDescriptor VertexBuffer::getAttributeDescriptor(int index) const{
		return layout[index];
	}
	
}