#pragma once
#include "core/definition/CoreExports.h"
#include "engine/base/Structures.h"
#include <stack>

namespace Ifrit::Engine::BufferManager {
	class TrivialBufferManager;

	struct IfritBuffer {
		int id = -1;
		std::weak_ptr<TrivialBufferManager> manager;
	};

	struct IfritBufferMetadata {
		std::unique_ptr<char[]> data = nullptr;
		size_t size = 0;
		bool maintained = false;
	};

	namespace Impl {
		class BufferManagerImpl {
		private:
			std::weak_ptr<TrivialBufferManager> wrapperObject;
			std::vector<IfritBuffer> buffers;
			std::vector<IfritBufferMetadata> bufferMetadata;
			std::stack<int> freeBufferIds;
		protected:
			int allocateBufferId();

		public:
			BufferManagerImpl(std::shared_ptr<TrivialBufferManager> wrapperObject);
			~BufferManagerImpl();

			IfritBuffer createBuffer(const IfritBufferCreateInfo& pCI);
			void destroyBuffer(const IfritBuffer& buffer);
			void mapBufferMemory(const IfritBuffer& buffer, void** ppData);
			void bufferData(const IfritBuffer& buffer, const void* src, size_t offset, size_t size);
			void bufferDataUnsafe(const IfritBuffer& buffer, const void* src, size_t offset, size_t size) IFRIT_AP_NOTHROW;
		};
	}

	class IFRIT_APIDECL TrivialBufferManager : public std::enable_shared_from_this<TrivialBufferManager> {
	private:
		bool initialized = false;
		std::unique_ptr<Impl::BufferManagerImpl> impl;
	public:
		TrivialBufferManager();
		~TrivialBufferManager();
		void init();
		IfritBuffer createBuffer(const IfritBufferCreateInfo& pCI);
		void destroyBuffer(const IfritBuffer& buffer);
		void mapBufferMemory(const IfritBuffer& buffer, void** ppData);
		void bufferData(const IfritBuffer& buffer, const void* src, size_t offset, size_t size);
	};
}