#include "engine/bufferman/BufferManager.h"

namespace Ifrit::Engine::BufferManager::Impl {
	int BufferManagerImpl::allocateBufferId(){
		if (freeBufferIds.empty()) {
			this->bufferMetadata.resize(this->bufferMetadata.size() + 1);
			this->buffers.resize(this->buffers.size() + 1);
			return this->buffers.size() - 1;
		}
		else {
			int id = freeBufferIds.top();
			freeBufferIds.pop();
			return id;
		}
	}
	BufferManagerImpl::BufferManagerImpl(std::shared_ptr<TrivialBufferManager> wrapperObject){
		this->wrapperObject = wrapperObject;
	}
	BufferManagerImpl::~BufferManagerImpl() = default;

	IfritBuffer BufferManagerImpl::createBuffer(const IfritBufferCreateInfo& pCI){
		auto bufferId = allocateBufferId();
		buffers[bufferId].id = bufferId;
		buffers[bufferId].manager = this->wrapperObject;
		bufferMetadata[bufferId].size = pCI.bufferSize;
		bufferMetadata[bufferId].data = std::make_unique<char[]>(pCI.bufferSize);
		bufferMetadata[bufferId].maintained = true;
		return buffers[bufferId];
	}

	void BufferManagerImpl::destroyBuffer(const IfritBuffer& buffer){
		auto maintained = bufferMetadata[buffer.id].maintained;
		if(maintained && buffer.id>=0 && !buffer.manager.owner_before(wrapperObject) && !wrapperObject.owner_before(buffer.manager)){
			freeBufferIds.push(buffer.id);
			bufferMetadata[buffer.id].size = 0;
			bufferMetadata[buffer.id].data = nullptr;
			bufferMetadata[buffer.id].maintained = false;
		}
		else {
			throw std::runtime_error("Buffer does not belong to this manager");
		}
	}

	void BufferManagerImpl::mapBufferMemory(const IfritBuffer& buffer, void** ppData){
		if(!bufferMetadata[buffer.id].maintained){
			throw std::runtime_error("Buffer is not maintained by this manager");
		}
		*ppData = bufferMetadata[buffer.id].data.get();
	}

	void BufferManagerImpl::bufferData(const IfritBuffer& buffer, const void* src, size_t offset, size_t size){
		if(!bufferMetadata[buffer.id].maintained){
			throw std::runtime_error("Buffer is not maintained by this manager");
		}
		if(offset + size > bufferMetadata[buffer.id].size){
			throw std::runtime_error("Buffer overflow");
		}
		memcpy(bufferMetadata[buffer.id].data.get() + offset, src, size);
	}



}

namespace Ifrit::Engine::BufferManager {
	TrivialBufferManager::TrivialBufferManager(){}
	TrivialBufferManager::~TrivialBufferManager() {}	
	void TrivialBufferManager::init() {
		initialized = true;
		impl = std::make_unique<Impl::BufferManagerImpl>(shared_from_this());
	}
	IfritBuffer TrivialBufferManager::createBuffer(const IfritBufferCreateInfo& pCI){
		if (!initialized) {
			ifritError("Buffer manager not initialized");
		}
		return impl->createBuffer(pCI);
	}
	void TrivialBufferManager::destroyBuffer(const IfritBuffer& buffer){
		if (!initialized) {
			ifritError("Buffer manager not initialized");
		}
		return impl->destroyBuffer(buffer);
	}
	void TrivialBufferManager::mapBufferMemory(const IfritBuffer& buffer, void** ppData){
		if (!initialized) {
			ifritError("Buffer manager not initialized");
		}
		return impl->mapBufferMemory(buffer, ppData);
	}
	void TrivialBufferManager::bufferData(const IfritBuffer& buffer, const void* src, size_t offset, size_t size){
		if (!initialized) {
			ifritError("Buffer manager not initialized");
		}
		return impl->bufferData(buffer, src, offset, size);
	}
}