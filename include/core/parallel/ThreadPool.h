#pragma once

#include "core/utility/CoreUtils.h"

namespace Ifrit::Core::Parallel {
	class ThreadPool {
	private:
		std::vector<std::thread> threads;
		std::queue<std::function<void()>> tasks;
		std::mutex mutex;
		std::condition_variable condition;
		bool stop;

	public:

	};
}