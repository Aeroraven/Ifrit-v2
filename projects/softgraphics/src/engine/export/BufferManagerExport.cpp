
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#include "ifrit/softgraphics/engine/export/BufferManagerExport.h"
#include "ifrit/softgraphics/engine/bufferman/BufferManager.h"

using namespace Ifrit::GraphicsBackend::SoftGraphics::BufferManager;

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