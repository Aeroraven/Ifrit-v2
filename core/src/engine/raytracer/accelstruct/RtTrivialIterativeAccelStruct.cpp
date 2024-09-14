#include "engine/raytracer/accelstruct/RtTrivialIterativeAccelStruct.h"
#include "math/VectorOps.h"
#include "math/simd/SimdVectors.h"
using namespace Ifrit::Math::SIMD;

namespace Ifrit::Engine::Raytracer {

	namespace Impl {
		class TrivialBottomLevelASImpl : public BufferredAccelerationStructure<ifloat3> {
		private:
			const std::vector<ifloat3>* rawData;
			std::vector<vfloat3> data;

		public:
			virtual void bufferData(const std::vector<ifloat3>& vecData) override {
				this->rawData = &vecData;
				data = std::vector<vfloat3>(vecData.size());
				for(int i = 0; i < vecData.size(); i++){
					data[i] = vfloat3(vecData[i].x, vecData[i].y, vecData[i].z);
				}
			}
			virtual RayHit queryIntersection(const RayInternal& ray , float tmin, float tmax) const override {
				const auto cnt = size();
				RayHit prop;
				prop.id = -1;
				float dist = std::numeric_limits<float>::max();
				for (int i = 0; i < cnt; i++) {
					auto p = rayElementIntersection(ray, i);
					if (p.id !=-1 && p.t < dist) {
						dist = p.t;
						prop = p;
					}
				}
				return prop;
			}
			virtual int size() const {
				return this->data.size() / 3;
			}
			virtual RayHit rayElementIntersection(const RayInternal& ray, int index) const {
				using namespace Ifrit::Math;
				RayHit proposal;
				auto v0 = (this->data)[index * 3];
				auto v1 = (this->data)[index * 3 + 1];
				auto v2 = (this->data)[index * 3 + 2];
				auto e1 = v1 - v0;
				auto e2 = v2 - v0;
				auto p = cross(ray.r, e2);
				float det = dot(e1, p);
				if (det > -1e-6 && det < 1e-6) {
					//printf("Reject Ray EL: %f %f %f | Det=%f \n", ray.o.x, ray.o.y, ray.o.z, det);
					proposal.id = -1;
					return proposal;
				}
				float invDet = 1 / det;
				auto t = ray.o - v0;
				float u = dot(t, p) * invDet;
				if (u < 0 || u > 1) {
					//printf("Reject Ray U: %f %f %f | U=%f InvDet=%f \n", ray.o.x, ray.o.y, ray.o.z, u, invDet);
					proposal.id = -1;
					return proposal;
				}
				auto q = cross(t, e1);
				float v = dot(ray.r, q) * invDet;
				if (v < 0 || u + v > 1) {
					//printf("Reject Ray V: %f %f %f | U=%f, V=%f \n", ray.o.x, ray.o.y, ray.o.z, u,v);
					proposal.id = -1;
					return proposal;
				}
				float dist = dot(e2, q) * invDet;
				//printf("Final Accept Ray V: %f %f %f | %f %lld\n", ray.o.x, ray.o.y, ray.o.z, dist, this);
				proposal.id = index;
				proposal.p = { u,v,1 - u - v };
				proposal.t = dist;
				return proposal;
			}
			virtual void buildAccelerationStructure() {

			}
		};

		class TrivialTopLevelASImpl :public BufferredAccelerationStructure<TrivialBottomLevelAS*> {

		private:
			const std::vector<TrivialBottomLevelAS*>* data;

		public:
			virtual void bufferData(const std::vector<TrivialBottomLevelAS*>& vecData) override {
				this->data = &vecData;
			}
			virtual RayHit queryIntersection(const RayInternal& ray, float tmin, float tmax) const override {
				RayHit prop;
				prop.id = -1;
				float dist = std::numeric_limits<float>::max();
				for (const auto& x : *this->data) {
					auto p = x->queryIntersection(ray,tmin,tmax);
					if (p.id != -1 && p.t < dist) {
						prop = p;
						dist = p.t;
					}
				}
				return prop;
			}
			virtual int size() const {
				return this->data->size();
			}
			virtual void buildAccelerationStructure() {
			}
		};
	}


	TrivialBottomLevelAS::TrivialBottomLevelAS(){
		this->impl = new Impl::TrivialBottomLevelASImpl();
	}

	void TrivialBottomLevelAS::bufferData(const std::vector<ifloat3>& data){
		this->impl->bufferData(data);
	}

	RayHit TrivialBottomLevelAS::queryIntersection(const RayInternal& ray, float tmin, float tmax) const{
		return this->impl->queryIntersection(ray,tmin,tmax);
	}

	void TrivialBottomLevelAS::buildAccelerationStructure(){
		this->impl->buildAccelerationStructure();
	}

	TrivialTopLevelAS::TrivialTopLevelAS(){
		this->impl = new Impl::TrivialTopLevelASImpl();
	}

	void TrivialTopLevelAS::bufferData(const std::vector<TrivialBottomLevelAS*>& data){
		this->impl->bufferData(data);
	}

	RayHit TrivialTopLevelAS::queryIntersection(const RayInternal& ray, float tmin, float tmax) const{
		return this->impl->queryIntersection(ray,tmin,tmax);
	}

	void TrivialTopLevelAS::buildAccelerationStructure(){
		this->impl->buildAccelerationStructure();
	}

}