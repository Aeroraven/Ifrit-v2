#include "engine/raytracer/accelstruct/RtBoundingVolumeHierarchy.h"
#include "math/VectorOps.h"
#include <queue>
#include <stack>

namespace Ifrit::Engine::Raytracer::Impl {
	struct BVHNode{
		BoundingBox bbox;
		std::unique_ptr<BVHNode> left = nullptr, right = nullptr;
		int elementSize = 0;
	};

	class BoundingVolumeHierarchyBase {
	protected:
		std::unique_ptr<BVHNode> root = nullptr;
		std::vector<BoundingBox> bboxes;
		std::vector<ifloat3> centers;
		std::vector<int> belonging;
		std::vector<int> indices;
		int curSize = 0;
		const int maxDepth = 32;

	public:
		virtual int size() const = 0;
		virtual RayHit rayElementIntersection(const Ray& ray, int index) const = 0;
		virtual BoundingBox getElementBbox(int index) const = 0;
		virtual ifloat3 getElementCenter(int index) const = 0;
		
		void buildBVH() {
			this->root = std::make_unique<BVHNode>();
			this->curSize = this->size();

			this->bboxes = std::vector<BoundingBox>(this->size());
			this->indices = std::vector<int>(this->size());
			this->centers = std::vector<ifloat3>(this->size());
			this->belonging = std::vector<int>(this->size());
			for (int i = 0; i < this->size(); i++) {
				this->bboxes[i] = this->getElementBbox(i);
				this->centers[i] = this->getElementCenter(i);
				this->belonging[i] = i;
			}
			this->root->elementSize = this->size();
			this->buildBVHNode();
		}

		inline float rayBoxIntersection(const Ray& ray, const BoundingBox& bbox) const {
			using namespace Ifrit::Math;
			float tmin = std::numeric_limits<float>::lowest();
			float tmax = std::numeric_limits<float>::max();
			if (fabs(ray.r.x) < 1e-10) {
				if(ray.o.x < bbox.bmin.x || ray.o.x > bbox.bmax.x) return -1;
			}
			else {
				float tx1 = (bbox.bmin.x - ray.o.x) / ray.r.x;
				float tx2 = (bbox.bmax.x - ray.o.x) / ray.r.x;
				tmin = std::max(tmin, std::min(tx1, tx2));
				tmax = std::min(tmax, std::max(tx1, tx2));
			}

			if (fabs(ray.r.y) < 1e-10) {
				if (ray.o.y < bbox.bmin.y || ray.o.y > bbox.bmax.y) return -1;
			}
			else {
				float ty1 = (bbox.bmin.y - ray.o.y) / ray.r.y;
				float ty2 = (bbox.bmax.y - ray.o.y) / ray.r.y;
				tmin = std::max(tmin, std::min(ty1, ty2));
				tmax = std::min(tmax, std::max(ty1, ty2));
			}

			if (fabs(ray.r.z) < 1e-10) {
				if (ray.o.z < bbox.bmin.z || ray.o.z > bbox.bmax.z) return -1;
			}
			else {
				float tz1 = (bbox.bmin.z - ray.o.z) / ray.r.z;
				float tz2 = (bbox.bmax.z - ray.o.z) / ray.r.z;
				tmin = std::max(tmin, std::min(tz1, tz2));
				tmax = std::min(tmax, std::max(tz1, tz2));
			}
			if (tmin > tmax) {
				return -1;
			}
			return tmin;
		}

		int findSplit(int start, int end, int axis,float mid) {
			using namespace Ifrit::Math;
			int l = start, r = end;

			while (l < r) {
				while (l < r && elementAt(this->centers[this->belonging[l]], axis) < mid) l++;
				while (l < r && elementAt(this->centers[this->belonging[r]], axis) >= mid) r--;
				if (l < r) {
					std::swap(this->belonging[l], this->belonging[r]);
					l++;
					r--;
				}
			}
			auto pivot = elementAt(this->centers[this->belonging[l]],axis) < mid ? l : l - 1;
			return pivot;
		}

		void buildBVHNode() {
			using namespace Ifrit::Math;
			std::queue<std::tuple<BVHNode*,int,int>> q;
			q.push({ this->root.get(),0,0 });

			while (!q.empty()) {
				ifloat3 largestBBox = ifloat3{ -std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max() };
				auto& [node,depth,start] = q.front();
				BoundingBox& bbox = node->bbox;
				bbox.bmax = ifloat3{ -std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max() };
				bbox.bmin = ifloat3{ std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max() };

				for (int i = 0; i < node->elementSize; i++) {
					bbox.bmax = max(bbox.bmax, this->bboxes[this->belonging[start + i]].bmax);
					bbox.bmin = min(bbox.bmin, this->bboxes[this->belonging[start + i]].bmin);
					largestBBox = max(largestBBox, this->bboxes[this->belonging[start + i]].bmax - this->bboxes[this->belonging[start + i]].bmin);
				}
				if (depth >= this->maxDepth || node->elementSize <= 1) {
					q.pop();
					continue;
				}

				ifloat3 diff = bbox.bmax - bbox.bmin;
				ifloat3 midv = (bbox.bmax + bbox.bmin) * 0.5f;
				int axis = 0;
				if (diff.y > diff.x) axis = 1;
				if (diff.z > diff.y && diff.z > diff.x) axis = 2;
				float midvp = elementAt(midv, axis);

				int pivot = this->findSplit(start, start + node->elementSize - 1, axis, midvp);
				node->left = std::make_unique<BVHNode>();
				node->right = std::make_unique<BVHNode>();
				node->left->elementSize = pivot - start + 1;
				node->right->elementSize = node->elementSize - node->left->elementSize;
				
				if (node->left->elementSize != 0 && node->right->elementSize != 0) {
					q.push({ node->left.get(),depth + 1,start });
					q.push({ node->right.get(),depth + 1,pivot + 1 });
				}
				else {
					node->left = nullptr;
					node->right = nullptr;
				}
				q.pop();
			}
			for (int i = 0; i < this->curSize; i++) {
				this->indices[this->belonging[i]] = i;
			}
		}

		RayHit queryRayIntersection(const Ray& ray) const {
			RayHit prop;
			prop.id = -1;
			prop.t = std::numeric_limits<float>::max();
			const auto nodeRoot = this->root.get();
			float rootHit = this->rayBoxIntersection(ray, nodeRoot->bbox);
			
			if (rootHit<0) return prop;
			std::stack<std::tuple<BVHNode*,int,float>> q;
			q.push({ nodeRoot,0, rootHit });
			float minDist = std::numeric_limits<float>::max();
			
			while (!q.empty()) {
				auto p = q.top();
				auto& [node,start,cmindist] = p;
				q.pop();
				
				if (cmindist > minDist) {
					continue;
				}
				float leftIntersect = -1, rightIntersect = -1;
				if (node->left) leftIntersect = this->rayBoxIntersection(ray, node->left->bbox);
				if (node->right) rightIntersect = this->rayBoxIntersection(ray, node->right->bbox);
				if (leftIntersect > 0 && rightIntersect > 0) {
					if (leftIntersect < rightIntersect) {
						q.push({ node->right.get(),start + node->left->elementSize,rightIntersect });
						q.push({ node->left.get(),start,leftIntersect });
					}
					else {
						q.push({ node->left.get(),start,leftIntersect });
						q.push({ node->right.get(), start + node->left->elementSize,rightIntersect });
					}
				}
				else if (leftIntersect > 0) {
					q.push({ node->left.get(),start,leftIntersect });
				}
				else if (rightIntersect > 0) {
					q.push({ node->right.get(), start + node->left->elementSize,rightIntersect });
				}
				else {
					if (node->left == nullptr && node->right == nullptr) {
						for (int i = 0; i < node->elementSize; i++) {
							int index = this->belonging[i + start];
							auto dist = this->rayElementIntersection(ray, index);
							if (dist.t >1e-9 && dist.t < minDist) {
								minDist = dist.t;
								prop = dist;
							}
						}
					}
				}
			}
			return prop;
		}
	};


	class BoundingVolumeHierarchyBottomLevelASImpl : public BoundingVolumeHierarchyBase, public BufferredAccelerationStructure<ifloat3> {
	private:
		const std::vector<ifloat3>* data;

	public:
		virtual void bufferData(const std::vector<ifloat3>& vecData) override {
			this->data = &vecData;
		}
		virtual RayHit queryIntersection(const Ray& ray) const override {
			return this->queryRayIntersection(ray);
		}
		virtual int size() const override {
			return this->data->size() / 3;
		}
		virtual RayHit rayElementIntersection(const Ray& ray, int index) const override {
			using namespace Ifrit::Math;
			RayHit proposal;
			proposal.id = -1;
			proposal.t = std::numeric_limits<float>::max();
			ifloat3 v0 = (*this->data)[index * 3];
			ifloat3 v1 = (*this->data)[index * 3 + 1];
			ifloat3 v2 = (*this->data)[index * 3 + 2];
			ifloat3 e1 = v1 - v0;
			ifloat3 e2 = v2 - v0;
			ifloat3 p = cross(ray.r, e2);
			float det = dot(e1, p);
			if (det > -1e-8 && det < 1e-8) {
				return proposal;
			}
			float invDet = 1 / det;
			ifloat3 t = ray.o - v0;
			float u = dot(t, p) * invDet;
			if (u < 0 || u > 1) {
				return proposal;
			}
			ifloat3 q = cross(t, e1);
			float v = dot(ray.r, q) * invDet;
			if (v < 0 || u + v > 1) {
				return proposal;
			}
			float dist = dot(e2, q) * invDet;
			proposal.id = index;
			proposal.p = { u,v,1 - u - v };
			proposal.t = dist;
			return proposal;
		}
		virtual BoundingBox getElementBbox(int index) const override {
			using namespace Ifrit::Math;
			ifloat3 v0 = (*this->data)[index * 3];
			ifloat3 v1 = (*this->data)[index * 3 + 1];
			ifloat3 v2 = (*this->data)[index * 3 + 2];
			BoundingBox bbox;
			bbox.bmin = min(min(v0, v1), v2);
			bbox.bmax = max(max(v0, v1), v2);
			return bbox;
		}
		virtual BoundingBox getRootBbox() {
			return this->root->bbox;
		}
		virtual ifloat3 getElementCenter(int index) const override {
			using namespace Ifrit::Math;
			
			/*ifloat3 v0 = (*this->data)[index * 3];
			ifloat3 v1 = (*this->data)[index * 3 + 1];
			ifloat3 v2 = (*this->data)[index * 3 + 2];
			return (v0 + v1 + v2) / 3.0f;*/
			using namespace Ifrit::Math;
			ifloat3 v0 = (*this->data)[index * 3];
			ifloat3 v1 = (*this->data)[index * 3 + 1];
			ifloat3 v2 = (*this->data)[index * 3 + 2];
			BoundingBox bbox;
			bbox.bmin = min(min(v0, v1), v2);
			bbox.bmax = max(max(v0, v1), v2);
			return (bbox.bmin + bbox.bmax) * 0.5f;
		}
		virtual void buildAccelerationStructure() {
			this->buildBVH();
		}
	};

	class BoundingVolumeHierarchyTopLevelASImpl : public BoundingVolumeHierarchyBase,
		public BufferredAccelerationStructure<BoundingVolumeHierarchyBottomLevelAS*> {

	private:
		const std::vector<BoundingVolumeHierarchyBottomLevelAS*>* data;

	public:
		virtual void bufferData(const std::vector<BoundingVolumeHierarchyBottomLevelAS*>& vecData) override {
			this->data = &vecData;
		}
		virtual RayHit queryIntersection(const Ray& ray) const override {
			auto pv = this->queryRayIntersection(ray);
			return pv;
		}
		virtual int size() const override {
			return this->data->size();
		}
		virtual RayHit rayElementIntersection(const Ray& ray, int index) const override {
			auto p = (*this->data)[index]->queryIntersection(ray);
			return p;
		}
		virtual BoundingBox getElementBbox(int index) const override {
			auto x = (*this->data)[index]->impl->getRootBbox();
			return x;
		}
		virtual ifloat3 getElementCenter(int index) const override {
			using namespace Ifrit::Math;
			auto bbox = (*this->data)[index]->impl->getRootBbox();
			auto cx = (bbox.bmax + bbox.bmin) * 0.5f;
			return cx;
		}
		virtual void buildAccelerationStructure() {
			this->buildBVH();
		}
	};
}

namespace Ifrit::Engine::Raytracer {
	BoundingVolumeHierarchyBottomLevelAS::BoundingVolumeHierarchyBottomLevelAS() {
		this->impl = new Impl::BoundingVolumeHierarchyBottomLevelASImpl();
	}

	BoundingVolumeHierarchyBottomLevelAS::~BoundingVolumeHierarchyBottomLevelAS(){
		delete this->impl;
	}

	void BoundingVolumeHierarchyBottomLevelAS::bufferData(const std::vector<ifloat3>& data) {
		this->impl->bufferData(data);
	}

	RayHit BoundingVolumeHierarchyBottomLevelAS::queryIntersection(const Ray& ray) const {
		return this->impl->queryIntersection(ray);
	}
	void BoundingVolumeHierarchyBottomLevelAS::buildAccelerationStructure() {
		this->impl->buildAccelerationStructure();
	}

	BoundingVolumeHierarchyTopLevelAS::BoundingVolumeHierarchyTopLevelAS() {
		this->impl = new Impl::BoundingVolumeHierarchyTopLevelASImpl();
	}

	BoundingVolumeHierarchyTopLevelAS::~BoundingVolumeHierarchyTopLevelAS(){
		delete this->impl;
	}

	void BoundingVolumeHierarchyTopLevelAS::bufferData(const std::vector<BoundingVolumeHierarchyBottomLevelAS*>& data) {
		this->impl->bufferData(data);
	}

	RayHit BoundingVolumeHierarchyTopLevelAS::queryIntersection(const Ray& ray) const {
		auto x = this->impl->queryIntersection(ray);
		return x;
	}
	void BoundingVolumeHierarchyTopLevelAS::buildAccelerationStructure() {
		this->impl->buildAccelerationStructure();
	}
}