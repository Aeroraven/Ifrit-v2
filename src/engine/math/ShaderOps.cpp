#include "engine/math/ShaderOps.h"

namespace Ifrit::Engine::Math::ShaderOps {
	ifloat4 multiply(const float4x4 a, const ifloat4 b){
		ifloat4 result;
		result.x = a[0][0] * b.x + a[0][1] * b.y + a[0][2] * b.z + a[0][3] * b.w;
		result.y = a[1][0] * b.x + a[1][1] * b.y + a[1][2] * b.z + a[1][3] * b.w;
		result.z = a[2][0] * b.x + a[2][1] * b.y + a[2][2] * b.z + a[2][3] * b.w;
		result.w = a[3][0] * b.x + a[3][1] * b.y + a[3][2] * b.z + a[3][3] * b.w;
		return result;
	}
	float4x4 lookAt(ifloat3 eye, ifloat3 center, ifloat3 up){
		ifloat3 f = normalize(sub(center,eye));
		ifloat3 s = normalize(cross(f, up));
		ifloat3 u = cross(s, f);
		float4x4 result;
		result[0][0] = s.x;
		result[0][1] = s.y;
		result[0][2] = s.z;
		result[0][3] = 0;
		result[1][0] = u.x;
		result[1][1] = u.y;
		result[1][2] = u.z;
		result[1][3] = 0;
		result[2][0] = f.x;
		result[2][1] = f.y;
		result[2][2] = f.z;
		result[2][3] = 0;
		result[3][0] = 0;
		result[3][1] = 0;
		result[3][2] = 0;
		result[3][3] = 1;
		float4x4 trans;
		for (int i = 0; i <= 3; i++) {
			trans[i][0] = 0;
			trans[i][1] = 0;
			trans[i][2] = 0;
			trans[i][3] = 0;
		}
		trans[0][3] = -eye.x;
		trans[1][3] = -eye.y;
		trans[2][3] = -eye.z;
		trans[3][3] = 1;
		trans[0][0] = 1;
		trans[1][1] = 1;
		trans[2][2] = 1;
		return multiply(result,trans);
	}
	float4x4 perspective(float fovy, float aspect, float zNear, float zFar){
		float4x4 result;
		float f = 1.0f / tan(fovy / 2.0f);
		float halfFovy = fovy / 2.0f;
		float nTop = zNear * tan(halfFovy);
		float nRight = nTop * aspect;
		float nLeft = -nRight;
		float nBottom = -nTop;
		result[0][0] = 2 * zNear /(nRight-nLeft);
		result[1][0] = 0;
		result[2][0] = 0;
		result[3][0] = 0;
		result[0][1] = 0;
		result[1][1] = 2*zNear/(nTop-nBottom);
		result[2][1] = 0;
		result[3][1] = 0;
		result[0][2] = 0;
		result[1][2] = 0;
		result[2][2] = (zFar ) / (zFar - zNear);
		result[3][2] = 1;
		result[0][3] = 0;
		result[1][3] = 0;
		result[2][3] = -( zFar * zNear) / (zFar - zNear);
		result[3][3] = 0;
		return result;
	}
	float4x4 multiply(const float4x4 a, const float4x4 b){
		float4x4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
			}
		}
		return result;
	}
	ifloat4 normalize(ifloat4 a){
		float length = sqrt(a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w);
		return { a.x / length, a.y / length, a.z / length, a.w / length };
	}
	ifloat3 cross(ifloat3 a, ifloat3 b){
		return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
	}
	ifloat3 normalize(ifloat3 a){
		float length = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
		return { a.x / length, a.y / length, a.z / length };
	}
	ifloat3 sub(ifloat3 a, ifloat3 b) {
		return { a.x - b.x, a.y - b.y, a.z - b.z };
	}
	float dot(ifloat3 a, ifloat3 b){
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	float4x4 transpose(float4x4 a){
		float4x4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result[i][j] = a[j][i];
			}
		}
		return result;
	}
}