#include "engine/raytracer/accelstruct/RtBoundingVolumeHierarchy.h"
#include "math/VectorOps.h"
#include <queue>
#include <malloc.h>

namespace Ifrit::Engine::Raytracer::Impl {
	enum BVHSplitType {
		BST_TRIVIAL,
		BST_SAH
	};
	struct BVHNode{
		BoundingBox bbox;
		std::unique_ptr<BVHNode> left = nullptr, right = nullptr;
		int elementSize = 0;
		int startPos = 0;
	};

	class BoundingVolumeHierarchyBase {
	protected:
		std::unique_ptr<BVHNode> root = nullptr;
		std::vector<BoundingBox> bboxes;
		std::vector<ifloat3> centers;
		std::vector<int> belonging;
		std::vector<int> indices;
		int curSize = 0;
		int curMaxDepth = 0;
		BVHSplitType splitType = BST_TRIVIAL;
		static constexpr int maxDepth = 32;

	public:
		virtual int size() const = 0;
		virtual RayHit rayElementIntersection(const RayInternal& ray, int index) const = 0;
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

		inline float rayBoxIntersection(const RayInternal& ray, const BoundingBox& bbox) const {
			using namespace Ifrit::Math;
			auto t1 = (bbox.bmin - ray.o) * ray.invr;
			auto t2 = (bbox.bmax - ray.o) * ray.invr;
			auto v1 = min(t1, t2);
			auto v2 = max(t1, t2);
			float tmin = std::max(v1.x, std::max(v1.y, v1.z));
			float tmax = std::min(v2.x, std::min(v2.y, v2.z));;
			if (tmin > tmax) return -1;
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
			int profNodes = 0;

			while (!q.empty()) {
				profNodes++;
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
				node->startPos = start;
				curMaxDepth = std::max(curMaxDepth, depth + 1);
				if (depth >= maxDepth|| node->elementSize <= 1) {
					q.pop();
					continue;
				}

				int pivot = 0;
				
				if (splitType == BST_TRIVIAL) {
					ifloat3 diff = bbox.bmax - bbox.bmin;
					ifloat3 midv = (bbox.bmax + bbox.bmin) * 0.5f;
					int axis = 0;
					if (diff.y > diff.x) axis = 1;
					if (diff.z > diff.y && diff.z > diff.x) axis = 2;
					float midvp = elementAt(midv, axis);
					pivot = this->findSplit(start, start + node->elementSize - 1, axis, midvp);
				}
				else if (splitType == BST_SAH) {
					ifloat3 diff = bbox.bmax - bbox.bmin;
					auto minCost = diff.x * diff.y * diff.z * 2.0 * node->elementSize;
					float bestPivot = 0.0;
					int bestAxis = -1;
					for (int axis = 0; axis < 3; axis++) {
						for (int i = 1; i < 12; i++) {
							auto midv = lerp(bbox.bmin, bbox.bmax, 1.0f * i / 12);
							float midvp = elementAt(midv, axis);
							pivot = this->findSplit(start, start + node->elementSize - 1, axis, midvp);

							BoundingBox bLeft, bRight;
							// Bounding boxes
							bLeft.bmax = ifloat3{ -std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max() };
							bLeft.bmin = ifloat3{ std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max() };
							bRight.bmax = ifloat3{ -std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max() };
							bRight.bmin = ifloat3{ std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max() };

							for (int j = start; j <= pivot; j++) {
								auto idx = this->belonging[j];
								bLeft.bmax = max(bLeft.bmax, this->bboxes[idx].bmax);
								bLeft.bmin = min(bLeft.bmin, this->bboxes[idx].bmin);
							}
							for (int j = pivot; j < start + node->elementSize; j++) {
								auto idx = this->belonging[j];
								bRight.bmax = max(bRight.bmax, this->bboxes[idx].bmax);
								bRight.bmin = min(bRight.bmin, this->bboxes[idx].bmin);
							}
							auto dLeft = bLeft.bmax - bLeft.bmin;
							auto dRight = bRight.bmax - bRight.bmin;
							auto spLeft = dLeft.x * dLeft.y * dLeft.z * 2.0f;
							auto spRight = dRight.x * dRight.y * dRight.z * 2.0f;
							auto cost = spLeft * (pivot - start + 1) + spRight * (node->elementSize - (pivot - start + 1));
							if (cost < minCost) {
								minCost = cost;
								bestAxis = axis;
								bestPivot = midvp;
							}
						}
					}
					
					if (bestAxis == -1) {
						pivot = -2;
					}
					else {
						pivot = this->findSplit(start, start + node->elementSize - 1, bestAxis, bestPivot);
					}
				}

				node->left = std::make_unique<BVHNode>();
				node->right = std::make_unique<BVHNode>();
				node->left->elementSize = pivot - start + 1;
				node->right->elementSize = node->elementSize - node->left->elementSize;

				if (node->left->elementSize > 1 && node->right->elementSize > 1) {
					
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
			ifritLog1("BVH Built, Total Nodes:", profNodes);
		}

		RayHit queryRayIntersectionFromTLAS(const RayInternal& ray) const IFRIT_AP_NOTHROW {
			return queryRayIntersection<true>(ray);
		}
		template<bool doRootBoxIgnore>
		RayHit queryRayIntersection(const RayInternal& ray) const IFRIT_AP_NOTHROW {
			using namespace Ifrit::Math;
			RayHit prop;
			ifloat3 invd = ifloat3{ 1.0f,1.0f,1.0f } / ray.r;
			prop.id = -1;
			prop.t = std::numeric_limits<float>::max();
			const auto nodeRoot = this->root.get();

			constexpr float ignoreVal = std::numeric_limits<float>::max() * 0.5f;
			float rootHit;
			
			if constexpr(doRootBoxIgnore) {
				rootHit = ignoreVal;
			}
			else {
				rootHit = this->rayBoxIntersection(ray, nodeRoot->bbox);
				if (rootHit < 0) return prop;
			}

			constexpr auto dfsStackSize = BoundingVolumeHierarchyBase::maxDepth * 2 + 1;
			std::tuple<BVHNode*, float> q[dfsStackSize];
			int qPos = 0;
			
			q[qPos++] = { nodeRoot, rootHit };
			float minDist = std::numeric_limits<float>::max();
			
			while (qPos) {
				auto p = q[--qPos];
				auto& [node,cmindist] = p;
				if (cmindist >= minDist) {
					continue;
				}
				float leftIntersect = -1, rightIntersect = -1;
				const auto nLeft = node->left.get();
				const auto nRight = node->right.get();
				const auto nSize = node->elementSize;
				const auto nStartPos = node->startPos;
				if (nLeft == nullptr || nRight == nullptr) {
					for (int i = 0; i < nSize; i++) {
						int index = this->belonging[i + nStartPos];
						auto dist = this->rayElementIntersection(ray, index);
						if (dist.t > 1e-9 && dist.t < minDist) {
							minDist = dist.t;
							prop = dist;
						}
					}
				}
				else {
					leftIntersect = this->rayBoxIntersection(ray, nLeft->bbox);
					rightIntersect = this->rayBoxIntersection(ray, nRight->bbox);
					if (leftIntersect > minDist) leftIntersect = -1;
					if (rightIntersect > minDist) rightIntersect = -1;
					if (leftIntersect > 0 && rightIntersect > 0) {
						if (leftIntersect < rightIntersect) {
							q[qPos++] = { nRight,rightIntersect };
							q[qPos++] = { nLeft,leftIntersect };
						}
						else {
							q[qPos++] = { nLeft,leftIntersect };
							q[qPos++] = { nRight,rightIntersect };
						}
					}
					else if (leftIntersect > 0) {
						q[qPos++] = { nLeft,leftIntersect };
					}
					else if (rightIntersect > 0) {
						q[qPos++] = { nRight,rightIntersect };
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
		friend class BoundingVolumeHierarchyTopLevelASImpl;
		virtual void bufferData(const std::vector<ifloat3>& vecData) override {
			this->data = &vecData;
		}
		virtual RayHit queryIntersection(const RayInternal& ray) const override {
			return this->queryRayIntersection<false>(ray);
		}
		virtual int size() const override {
			return this->data->size() / 3;
		}
		inline virtual RayHit rayElementIntersection(const RayInternal& ray, int index) const override final {
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
		virtual BoundingBox getElementBbox(int index) const override final {
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
		virtual ifloat3 getElementCenter(int index) const override final {
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
		virtual RayHit queryIntersection(const RayInternal& ray) const override {
			auto pv = this->queryRayIntersection<false>(ray);
			return pv;
		}
		virtual int size() const override {
			return this->data->size();
		}
		virtual RayHit rayElementIntersection(const RayInternal& ray, int index) const override {
			auto p = (*this->data)[index]->queryIntersectionFromTLAS(ray);
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

	RayHit BoundingVolumeHierarchyBottomLevelAS::queryIntersection(const RayInternal& ray) const {
		return this->impl->queryIntersection(ray);
	}
	RayHit BoundingVolumeHierarchyBottomLevelAS::queryIntersectionFromTLAS(const RayInternal& ray) const{
		return this->impl->queryRayIntersectionFromTLAS(ray);
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

	RayHit BoundingVolumeHierarchyTopLevelAS::queryIntersection(const RayInternal& ray) const {
		auto x = this->impl->queryIntersection(ray);
		return x;
	}
	void BoundingVolumeHierarchyTopLevelAS::buildAccelerationStructure() {
		this->impl->buildAccelerationStructure();
	}
}