#pragma once
#include "engine/base/Shaders.h"
#include "engine/base/RaytracerBase.h"
#include <stack>

namespace Ifrit::Engine::Raytracer {
	// v2
	struct RaytracerShaderStackElement {
		Ray ray;
		RayHit rayHit;
		void* payloadPtr;
	};

	class RaytracerShaderExecutionStack {
	protected:
		std::vector<RaytracerShaderStackElement> execStack;
	public:
		IFRIT_HOST void pushStack(const Ray& ray,const RayHit& rayHit, void* pPayload);
		IFRIT_HOST void popStack();
		IFRIT_HOST virtual void onStackPushComplete() = 0;
		IFRIT_HOST virtual void onStackPopComplete() = 0;
	};

	class IFRIT_APIDECL RayGenShader : public ShaderBase {
	public:
		IFRIT_DUAL virtual void execute(
			const iint3& inputInvocation,
			const iint3& dimension,
			void* context
		) = 0;
		IFRIT_DUAL virtual ~RayGenShader() = default;
		IFRIT_HOST virtual RayGenShader* getCudaClone() { return nullptr; };
		IFRIT_HOST virtual std::unique_ptr<RayGenShader> getThreadLocalCopy() = 0;
		IFRIT_HOST virtual void updateUniformData(int binding, int set, const void* pData) {}
		IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return{}; }
	};

	class IFRIT_APIDECL MissShader : public ShaderBase, public RaytracerShaderExecutionStack {
	public:
		IFRIT_DUAL virtual void execute(void* context) = 0;
		IFRIT_DUAL virtual ~MissShader() = default;
		IFRIT_HOST virtual MissShader* getCudaClone() { return nullptr; };
		IFRIT_HOST virtual std::unique_ptr<MissShader> getThreadLocalCopy() = 0;
		IFRIT_HOST virtual void updateUniformData(int binding, int set, const void* pData) {}
		IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return{}; }
	};

	class IFRIT_APIDECL CloseHitShader : public ShaderBase, public RaytracerShaderExecutionStack {
	public:
		IFRIT_DUAL virtual void execute(
			const RayHit& hitAttribute,
			const Ray& ray,
			void* context
		) = 0;
		IFRIT_DUAL virtual ~CloseHitShader() = default;
		IFRIT_HOST virtual CloseHitShader* getCudaClone() { return nullptr; };
		IFRIT_HOST virtual std::unique_ptr<CloseHitShader> getThreadLocalCopy() = 0;
		IFRIT_HOST virtual void updateUniformData(int binding, int set, const void* pData) {};
		IFRIT_HOST virtual std::vector<std::pair<int, int>> getUniformList() { return{}; }
	};

	class IFRIT_APIDECL CallableShader : public ShaderBase, public RaytracerShaderExecutionStack {
	public:
		IFRIT_DUAL virtual void execute(
			void* outPayload,
			void* inPayload,
			void* context
		) = 0;
		IFRIT_HOST virtual CallableShader* getCudaClone() { return nullptr; };
	};

}