#pragma once
#include "../definition/CoreDefs.h"

namespace Ifrit::Core::Utility {

    template<typename T>
    concept CoreUtilIsIntegerType = std::is_integral_v<T>;

    template<typename... Args>
    class CoreUtilZipContainer {
    private:
        template <typename T>
        using ContainerIteratorTp = decltype(std::begin(std::declval<T&>()));
        template <typename T>
        using ContainerIteratorValTp = std::iterator_traits<ContainerIteratorTp<T>>::value_type;

        using GroupIteratorTp = std::tuple<ContainerIteratorTp<Args>...>;
        using GroupIteratorValTp = std::tuple<ContainerIteratorValTp<Args>...>;

        std::unique_ptr<std::tuple<Args...>> tuples;
    public:
        class iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = GroupIteratorValTp;
            using reference = const GroupIteratorValTp&;
            using pointer = GroupIteratorValTp*;

        private:
            std::unique_ptr<GroupIteratorTp> curIters;

        public:
            iterator(const GroupIteratorTp& iterators) {
                this->curIters = std::make_unique<GroupIteratorTp>(iterators);
            }
            iterator(const iterator& other) {
                GroupIteratorTp pIters = *other.curIters;
                this->curIters = std::make_unique<GroupIteratorTp>(pIters);
            }
            iterator& operator++() {
                std::apply([](auto&... p) {(++p, ...); }, *(this->curIters));
                return *this;
            }
            iterator operator++(int) {
                iterator copyIter(*this);
                std::apply([](auto&... p) {(++p, ...); }, *(this->curIters));
                return copyIter;
            }
            bool operator==(iterator p) const {
                return *curIters == *p.curIters;
            }
            bool operator!=(iterator p) const {
                return *curIters != *p.curIters;
            }
            GroupIteratorValTp operator*() const {
                return std::apply([](auto&... p) {return std::make_tuple(*p...); }, *(this->curIters));
            }
        };
    public:
        CoreUtilZipContainer(Args... args) {
            this->tuples = std::make_unique<std::tuple<Args...>>(std::make_tuple(args...));
        }
        iterator begin() {
            return iterator(std::apply([](auto&... p) { return std::make_tuple((p.begin())...); }, *(this->tuples)));
        }
        iterator end() {
            return iterator(std::apply([](auto&... p) { return std::make_tuple((p.end())...); }, *(this->tuples)));
        }

    };
    template<typename T>
    requires CoreUtilIsIntegerType<T>
    class CoreUtilRangedInterval {
    public:
        class iterator {
        public:
            using iterator_category = std::input_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = T;
            using reference = const T&;
            using pointer = T*;

        private:
            T curVal;
		    T step;
            T end;

        public:
            iterator(T start, T end, T step) {
				this->curVal = start;
				this->end = end;
				this->step = step;
			}
            iterator(const iterator& other) {
				this->curVal = other.curVal;
				this->end = other.end;
				this->step = other.step;
			}
            iterator& operator++() {
				this->curVal += this->step;
				return *this;
			}
            iterator operator++(int) {
				iterator copyIter(*this);
				this->curVal += this->step;
				return copyIter;
			}
            bool operator==(iterator p) const {
				return std::min(this->curVal,this->end) == p.curVal;
			}
            bool operator!=(iterator p) const {
				return std::min(this->curVal, this->end) != p.curVal;
			}
            T operator*() const {
				return this->curVal;
			}
        };
    private:
        T start;
        T endv;
        T step = 1;
    public:
        CoreUtilRangedInterval(T start, T end, T step) {
            this->start = start;
			this->endv = end;
			this->step = step;
        }
        CoreUtilRangedInterval(T start, T end) {
			this->start = start;
            this->endv = end;
            this->step = 1;
        }
        CoreUtilRangedInterval(T end) {
			this->start = 0;
			this->endv = end;
			this->step = 1;
		}
        
        iterator begin() {
			return iterator(this->start, this->endv, this->step);
		}

        iterator end() {
            return iterator(this->endv, this->endv, this->step);
        }

        size_t size() {
			return (this->endv - this->start) / this->step;
		}
    };

    template<typename... Args>
    void CoreUtilPrint(Args... args) {
        ((std::cout << args << " "), ...);
    }


#ifdef zip
    static_assert(false, "zip is already defined");
#endif
#ifdef range
    static_assert(false, "range is already defined");
#endif
#ifdef len
	static_assert(false, "len is already defined");
#endif


#define zip(...) Ifrit::Core::Utility::CoreUtilZipContainer(__VA_ARGS__)
#define range(...) Ifrit::Core::Utility::CoreUtilRangedInterval(__VA_ARGS__)
#define len(p) p.size()
#define printfmt(...) std::print(__VA_ARGS__)
#define prints(...) Ifrit::Core::Utility::CoreUtilPrint(__VA_ARGS__)
#define enumerate(p) zip(range(static_cast<decltype(len(p))>(0), len(p)), p)

}