#include "engine/export/BufferManagerExport.h"
#include "engine/bufferman/BufferManager.h"

using namespace Ifrit::Engine::GraphicsBackend::SoftGraphics::BufferManager;

struct TrivialBufferManagerWrapper {
  std::shared_ptr<TrivialBufferManager> manager;
  TrivialBufferManagerWrapper() {
    manager = std::make_shared<TrivialBufferManager>();
  }
  ~TrivialBufferManagerWrapper() = default;
};

struct IfritBufferWrapper {
  IfritBuffer buf;
};

IFRIT_APIDECL_COMPAT void *IFRIT_APICALL ifbufCreateBufferManager()
    IFRIT_EXPORT_COMPAT_NOTHROW {
  auto p = new TrivialBufferManagerWrapper();
  p->manager->init();
  return p;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifbufDestroyBufferManager(void *p)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  delete static_cast<TrivialBufferManagerWrapper *>(p);
}
IFRIT_APIDECL_COMPAT void *IFRIT_APICALL
ifbufCreateBuffer(void *pManager, size_t bufSize) IFRIT_EXPORT_COMPAT_NOTHROW {
  auto manager = static_cast<TrivialBufferManagerWrapper *>(pManager);
  auto buffer = manager->manager->createBuffer({bufSize});
  return new IfritBufferWrapper({buffer});
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL ifbufDestroyBuffer(void *p)
    IFRIT_EXPORT_COMPAT_NOTHROW {
  auto buffer = static_cast<IfritBufferWrapper *>(p);
  auto manager = buffer->buf.manager.lock();
  if (manager) {
    manager->destroyBuffer(buffer->buf);
  }
  delete buffer;
}
IFRIT_APIDECL_COMPAT void IFRIT_APICALL
ifbufBufferData(void *pBuffer, const void *pData, size_t offset,
                size_t size) IFRIT_EXPORT_COMPAT_NOTHROW {
  auto buffer = static_cast<IfritBufferWrapper *>(pBuffer);
  auto manager = buffer->buf.manager.lock();
  if (manager) {
    manager->bufferData(buffer->buf, pData, offset, size);
  }
}