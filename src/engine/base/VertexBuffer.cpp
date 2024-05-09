#include "engine/base/VertexBuffer.h"


namespace Ifrit::Engine {
	void VertexBuffer::setLayout(const std::vector<BufferLayout>& layout) {
		this->layout = layout;
		this->offsets.resize(layout.size());
		int offset = 0;
		for (int i = 0; i < layout.size(); i++) {
			offsets[i] = offset;
			offset += layout[i].size;
		}
	}

	void VertexBuffer::setVertexCount(const int vertexCount){
		this->vertexCount = vertexCount;
	}

	int VertexBuffer::getVertexCount() const{
		return vertexCount;
	}
	
}