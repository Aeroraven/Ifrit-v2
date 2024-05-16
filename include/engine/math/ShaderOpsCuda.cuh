#include "./core/definition/CoreTypes.h"
#include "./core/definition/CoreDefs.h"

namespace Ifrit::Engine::Math::ShaderOps::CUDA {
	IFRIT_DUAL inline float dot(const ifloat4& a, const ifloat4& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}
	IFRIT_DUAL inline ifloat4 sub(const ifloat4& a, const ifloat4& b) {
		return ifloat4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}
	IFRIT_DUAL inline ifloat4 multiply(const ifloat4& a, const float& b) {
		return ifloat4(a.x * b, a.y * b, a.z * b, a.w * b);
	}
	IFRIT_DUAL inline ifloat4 add(const ifloat4& a, const ifloat4& b) {
		return ifloat4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}
	IFRIT_DUAL inline ifloat4 lerp(const ifloat4& a, const ifloat4& b, const float& t) {
		return add(multiply(a, 1 - t), multiply(b, t));
	}
}