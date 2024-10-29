#pragma once
#include "ifrit/common/math/VectorOps.h"
template <class Archive> void serialize(Archive &ar, ifloat2 &v) {
  ar(v.x, v.y);
}
template <class Archive> void serialize(Archive &ar, ifloat3 &v) {
  ar(v.x, v.y, v.z);
}
template <class Archive> void serialize(Archive &ar, ifloat4 &v) {
  ar(v.x, v.y, v.z, v.w);
}