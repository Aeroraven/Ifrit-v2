#include "core/definition/CoreExports.h"

namespace Ifrit::Engine::Math::ShaderOps {
	float4 multiply(float4x4 a, float4 b);
	float4x4 lookAt(float3 eye, float3 center, float3 up);
	float4x4 perspective(float fovy, float aspect, float zNear, float zFar);
	float4x4 multiply(float4x4 a, float4x4 b);
	float4 normalize(float4 a);
	float3 cross(float3 a, float3 b);
	float3 normalize(float3 a);
	float3 sub(float3 a, float3 b);
	float dot(float3 a, float3 b);
	float4x4 transpose(float4x4 a);
}