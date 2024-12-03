
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


#pragma once
#include "../definition/CoreDefs.h"

namespace Ifrit::GraphicsBackend::SoftGraphics::Core::Utility {

template <typename T>
concept CoreUtilIsIntegerType = std::is_integral_v<T>;

template <typename... Args> class CoreUtilZipContainer {
private:
  template <typename T>
  using ContainerIteratorTp = decltype(std::begin(std::declval<T &>()));
  template <typename T>
  using ContainerIteratorValTp =
      std::iterator_traits<ContainerIteratorTp<T>>::value_type;

  using GroupIteratorTp = std::tuple<ContainerIteratorTp<Args>...>;
  using GroupIteratorValTp = std::tuple<ContainerIteratorValTp<Args>...>;

  std::unique_ptr<std::tuple<Args...>> tuples;

public:
  class iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = GroupIteratorValTp;
    using reference = const GroupIteratorValTp &;
    using pointer = GroupIteratorValTp *;

  private:
    std::unique_ptr<GroupIteratorTp> curIters;

  public:
    iterator(const GroupIteratorTp &iterators) {
      this->curIters = std::make_unique<GroupIteratorTp>(iterators);
    }
    iterator(const iterator &other) {
      GroupIteratorTp pIters = *other.curIters;
      this->curIters = std::make_unique<GroupIteratorTp>(pIters);
    }
    iterator &operator++() {
      std::apply([](auto &...p) { (++p, ...); }, *(this->curIters));
      return *this;
    }
    iterator operator++(int) {
      iterator copyIter(*this);
      std::apply([](auto &...p) { (++p, ...); }, *(this->curIters));
      return copyIter;
    }
    bool operator==(iterator p) const { return *curIters == *p.curIters; }
    bool operator!=(iterator p) const { return *curIters != *p.curIters; }
    GroupIteratorValTp operator*() const {
      return std::apply([](auto &...p) { return std::make_tuple(*p...); },
                        *(this->curIters));
    }
  };

public:
  CoreUtilZipContainer(Args... args) {
    this->tuples =
        std::make_unique<std::tuple<Args...>>(std::make_tuple(args...));
  }
  iterator begin() {
    return iterator(
        std::apply([](auto &...p) { return std::make_tuple((p.begin())...); },
                   *(this->tuples)));
  }
  iterator end() {
    return iterator(
        std::apply([](auto &...p) { return std::make_tuple((p.end())...); },
                   *(this->tuples)));
  }
};
template <typename T>
  requires CoreUtilIsIntegerType<T>
class CoreUtilRangedInterval {
public:
  class iterator {
  public:
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using reference = const T &;
    using pointer = T *;

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
    iterator(const iterator &other) {
      this->curVal = other.curVal;
      this->end = other.end;
      this->step = other.step;
    }
    iterator &operator++() {
      this->curVal += this->step;
      return *this;
    }
    iterator operator++(int) {
      iterator copyIter(*this);
      this->curVal += this->step;
      return copyIter;
    }
    bool operator==(iterator p) const {
      return std::min(this->curVal, this->end) == p.curVal;
    }
    bool operator!=(iterator p) const {
      return std::min(this->curVal, this->end) != p.curVal;
    }
    T operator*() const { return this->curVal; }
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

  iterator begin() { return iterator(this->start, this->endv, this->step); }

  iterator end() { return iterator(this->endv, this->endv, this->step); }

  size_t size() { return (this->endv - this->start) / this->step; }
};

template <typename... Args> void CoreUtilPrint(Args... args) {
  ((std::cout << args << " "), ...);
}

inline uint32_t byteSwapU32(uint32_t x) {
#ifdef _MSC_VER
  return _byteswap_ulong(x);
#else
  return __builtin_bswap32(x);
#endif
}

inline int byteSwapI32(int x) {
  auto ux = *(unsigned int *)(&x);
  auto fx = byteSwapU32(ux);
  return *(int *)(&fx);
}

struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &p) const {
    // https://stackoverflow.com/questions/32685540/why-cant-i-compile-an-unordered-map-with-a-pair-as-key
    auto h1 = std::hash<T1>{}(p.first);
    auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

} // namespace Ifrit::GraphicsBackend::SoftGraphics::Core::Utility