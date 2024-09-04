#pragma once
#include "./core/definition/CoreTypes.h"
#include "./core/definition/CoreDefs.h"

namespace Ifrit::Engine::Math::ShaderOps {
	IFRIT_APIDECL float4x4 lookAt(ifloat3 eye, ifloat3 center, ifloat3 up);
	IFRIT_APIDECL float4x4 perspective(float fovy, float aspect, float zNear, float zFar);
	IFRIT_APIDECL ifloat4 multiply(const float4x4 a, const ifloat4 b);
	IFRIT_APIDECL float4x4 multiply(const float4x4 a, const float4x4 b);
	IFRIT_APIDECL float4x4 transpose(const float4x4& a);

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