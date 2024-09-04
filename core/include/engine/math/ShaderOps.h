#pragma once
#include "./core/definition/CoreTypes.h"
#include "./core/definition/CoreDefs.h"

namespace Ifrit::Engine::Math::ShaderOps {
	IFRIT_APIDECL ifloat4 multiply(const float4x4 a, const ifloat4 b);
	IFRIT_APIDECL float4x4 lookAt(ifloat3 eye, ifloat3 center, ifloat3 up);
	IFRIT_APIDECL float4x4 perspective(float fovy, float aspect, float zNear, float zFar);
	IFRIT_APIDECL float4x4 multiply(const float4x4 a, const float4x4 b);
	IFRIT_APIDECL ifloat4 normalize(ifloat4 a);
	IFRIT_APIDECL ifloat3 cross(ifloat3 a, ifloat3 b);
	IFRIT_APIDECL ifloat3 normalize(ifloat3 a);
	IFRIT_APIDECL ifloat2 normalize(ifloat2 a);
	IFRIT_APIDECL ifloat3 sub(ifloat3 a, ifloat3 b);
	IFRIT_APIDECL float dot(ifloat3 a, ifloat3 b);
	IFRIT_APIDECL float4x4 transpose(float4x4 a);

	inline float dot(const ifloat4& a, const ifloat4& b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}
	inline ifloat4 sub(const ifloat4& a, const ifloat4& b) {
		return ifloat4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}
	inline ifloat4 multiply(const ifloat4& a, const float& b) {
		return ifloat4(a.x * b, a.y * b, a.z * b, a.w * b);
	}
	inline ifloat4 add(const ifloat4& a, const ifloat4& b) {
		return ifloat4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}
	inline ifloat4 lerp(const ifloat4& a, const ifloat4& b, const float& t) {
		return add(multiply(a, 1 - t), multiply(b, t));
	}
	inline float4x4 axisAngleRotation(ifloat3 axis, float angle) {
		float c = cos(angle);
		float s = sin(angle);
		float t = 1 - c;
		float x = axis.x;
		float y = axis.y;
		float z = axis.z;
		float4x4 ret;
		ret[0][0] = t * x * x + c;
		ret[0][1] = t * x * y + s * z;
		ret[0][2] = t * x * z - s * y;
		ret[0][3] = 0;
		ret[1][0] = t * x * y - s * z;
		ret[1][1] = t * y * y + c;
		ret[1][2] = t * y * z + s * x;
		ret[1][3] = 0;
		ret[2][0] = t * x * z + s * y;
		ret[2][1] = t * y * z - s * x;
		ret[2][2] = t * z * z + c;
		ret[2][3] = 0;
		ret[3][0] = 0;
		ret[3][1] = 0;
		ret[3][2] = 0;
		ret[3][3] = 1;
		return ret;
	}
}