#include "engine/raytracer/RtShaders.h"

namespace Ifrit::Engine::Raytracer {
	IFRIT_HOST void RaytracerShaderExecutionStack::pushStack(const Ray& ray, const RayHit& rayHit, void* pPayload){
		RaytracerShaderStackElement el;
		el.ray = ray;
		el.payloadPtr = pPayload;
		el.rayHit = rayHit;
		execStack.push(el);
		this->onStackPushComplete();
	}
	IFRIT_HOST void RaytracerShaderExecutionStack::popStack(){
		execStack.pop();
		this->onStackPopComplete();
	}
}