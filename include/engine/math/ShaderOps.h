#include "core/definition/CoreExports.h"

namespace Ifrit::Engine::Math::ShaderOps {
	float4 multiply(const float4x4 a, const float4 b);
	float4x4 lookAt(float3 eye, float3 center, float3 up);
	float4x4 perspective(float fovy, float aspect, float zNear, float zFar);
	float4x4 multiply(const float4x4 a, const float4x4 b);
	float4 normalize(float4 a);
	float3 cross(float3 a, float3 b);
	float3 normalize(float3 a);
	float3 sub(float3 a, float3 b);
	float dot(float3 a, float3 b);
	float4x4 transpose(float4x4 a);

	inline float dot(float4 a, float4 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}
	inline float4 sub(float4 a, float4 b) {
		return float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
	}
	inline float4 multiply(float4 a, float b) {
		return float4(a.x * b, a.y * b, a.z * b, a.w * b);
	}
	inline float4 add(float4 a, float4 b) {
		return float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
	}
	inline float4 lerp(float4 a, float4 b, float t) {
		return add(multiply(a, 1 - t), multiply(b, t));
	}
}