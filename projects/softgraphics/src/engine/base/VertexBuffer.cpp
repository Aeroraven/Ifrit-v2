#include "ifrit/softgraphics/engine/base/VertexBuffer.h"
#include "ifrit/softgraphics/engine/base/TypeDescriptor.h"

namespace Ifrit::GraphicsBackend::SoftGraphics {
IFRIT_APIDECL VertexBuffer::VertexBuffer() {
  this->context = new std::remove_pointer_t<decltype(this->context)>();
}
IFRIT_APIDECL VertexBuffer::~VertexBuffer() { delete this->context; }

IFRIT_APIDECL void VertexBuffer::allocateBuffer(const size_t numVertices) {
  int elementSizeX = 0;
  for (int i = 0; i < context->layout.size(); i++) {
    elementSizeX += context->layout[i].size;
  }
  context->buffer.resize(numVertices * elementSizeX);
  this->vertexCount = static_cast<int>(numVertices);
}

IFRIT_APIDECL void
VertexBuffer::setLayout(const std::vector<TypeDescriptor> &layout) {
  this->context->layout = layout;
  this->context->offsets.resize(layout.size());
  int offset = 0;
  for (int i = 0; i < layout.size(); i++) {
    context->offsets[i] = offset;
    offset += layout[i].size;
    if (layout[i].type == TypeDescriptorEnum::IFTP_UNDEFINED) {
      printf("Undefined layout %d\n", layout[i].type);
      std::abort();
    }
  }
  elementSize = offset;
}

IFRIT_APIDECL void VertexBuffer::setVertexCount(const int vcnt) {
  this->vertexCount = vcnt;
}

IFRIT_APIDECL int VertexBuffer::getVertexCount() const { return vertexCount; }

IFRIT_APIDECL int VertexBuffer::getAttributeCount() const {
  return static_cast<int>(context->layout.size());
}

IFRIT_APIDECL TypeDescriptor
VertexBuffer::getAttributeDescriptor(int index) const {
  return context->layout[index];
}

/* DLL Compatible */
IFRIT_APIDECL void
VertexBuffer::setLayoutCompatible(const TypeDescriptor *layouts, int num) {
  std::vector<TypeDescriptor> clayouts(num);
  for (int i = 0; i < num; i++) {
    clayouts[i] = layouts[i];
  }
  setLayout(clayouts);
}
IFRIT_APIDECL void VertexBuffer::setValueFloat4Compatible(const int index,
                                                          const int attribute,
                                                          const ifloat4 value) {
  this->setValue<ifloat4>(index, attribute, value);
}

} // namespace Ifrit::GraphicsBackend::SoftGraphics