
/*
Ifrit-v2
Copyright (C) 2024 funkybirds(Aeroraven)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once
#include "AssetReference.h"
#include "ifrit/common/logging/Logging.h"
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/common/util/TypingUtil.h"
#include "ifrit/rhi/common/RhiLayer.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define IFRIT_COMPONENT_SERIALIZE(...)                                         \
  IFRIT_STRUCT_SERIALIZE(m_id, m_parentObject, __VA_ARGS__)
#define IFRIT_COMPONENT_REGISTER(x)                                            \
  IFRIT_DERIVED_REGISTER(x);                                                   \
  IFRIT_INHERIT_REGISTER(Ifrit::Core::Component, x);

namespace Ifrit::Core {

struct ComponentIdentifier {
  std::string m_uuid;
  std::string m_name;

  IFRIT_STRUCT_SERIALIZE(m_uuid, m_name)
};

template <class T> class AttributeOwner {
protected:
  T m_attributes{};

public:
  inline std::string serializeAttribute() {
    std::string serialized;
    Ifrit::Common::Serialization::serialize(m_attributes, serialized);
    return serialized;
  }
  inline void deserializeAttribute() {
    std::string serialized;
    Ifrit::Common::Serialization::deserialize(serialized, m_attributes);
  }
};

class Component;
class SceneObject;
class Transform;

// TODO: for performance considerations, components container is not consistent
// across different build envs.

class IFRIT_APIDECL SceneObject
    : public Ifrit::Common::Utility::NonCopyable,
      public std::enable_shared_from_this<SceneObject> {
protected:
  ComponentIdentifier m_id;
  std::string m_name;
  // std::unordered_map<std::string, std::shared_ptr<Component>> m_components;
  std::vector<std::shared_ptr<Component>> m_components;
  std::unordered_map<std::string, uint32_t> m_componentIndex;
  std::unordered_map<size_t, uint32_t> m_componentsHashed;

public:
  SceneObject();
  virtual ~SceneObject() = default;
  void initialize();

  static std::shared_ptr<SceneObject> createPrefab();

  template <class T> void addComponent(std::shared_ptr<T> component) {
    static_assert(std::is_base_of<Component, T>::value,
                  "T must be derived from Component");
    auto typeName = typeid(T).name();
    auto typeHash = typeid(T).hash_code();
    if (m_componentIndex.count(typeName) > 0) {
      iError("Component already exists");
      std::abort();
    }
    if (m_componentsHashed.count(typeHash) > 0) {
      iError("Hash conflicted");
      std::abort();
    }
    m_components.push_back(component);
    m_componentIndex[typeName] = m_components.size() - 1;
    m_componentsHashed[typeHash] = m_components.size() - 1;
  }

  template <class T> std::shared_ptr<T> addComponent() {
    static_assert(std::is_base_of<Component, T>::value,
                  "T must be derived from Component");
    auto typeName = typeid(T).name();
    auto typeHash = typeid(T).hash_code();
    if (m_componentIndex.count(typeName) > 0) {
      auto idx = m_componentIndex[typeName];
      return std::static_pointer_cast<T>(m_components[idx]);
    }
    if (m_componentsHashed.count(typeHash) > 0) {
      iError("Hash conflicted");
      std::abort();
    }
    auto ret = std::make_shared<T>(shared_from_this());
    m_components.push_back(ret);
    m_componentIndex[typeName] = m_components.size() - 1;
    m_componentsHashed[typeHash] = m_components.size() - 1;
    return ret;
  }

  template <class T> std::shared_ptr<T> getComponent() {
    static_assert(std::is_base_of<Component, T>::value,
                  "T must be derived from Component");
    auto typeHash = typeid(T).hash_code();
    if (m_componentsHashed.count(typeHash) == 0) {
      return nullptr;
    }
    auto itIndex = m_componentsHashed[typeHash];
#ifdef _DEBUG
    auto ret = std::dynamic_pointer_cast<T>(m_components[itIndex]);
    if (ret == nullptr) {
      iError("Invalid cast");
      std::abort();
    }
    return ret;
#else
    return std::static_pointer_cast<T>(m_components[itIndex]);
#endif
  }

  // Unsafe version of getComponent, use with caution. It's intended to be used
  // in performance-critical code.
  template <class T> T *getComponentUnsafe() {
    static_assert(std::is_base_of<Component, T>::value,
                  "T must be derived from Component");
    auto typeHash = typeid(T).hash_code();
    if (m_componentsHashed.count(typeHash) == 0) {
      return nullptr;
    }
    auto itIndex = m_componentsHashed[typeHash];
    return static_cast<T *>(m_components[itIndex].get());
  }

  inline std::vector<std::shared_ptr<Component>> getAllComponents() {
    return m_components;
  }

  inline void setName(const std::string &name) { m_name = name; }
  IFRIT_STRUCT_SERIALIZE(m_id, m_name, m_components, m_componentIndex,
                         m_componentsHashed);
};

// Here, SceneObjectPrefab is a type alias for SceneObject.
// It only indiecates that the object does not belong to the scene.
using SceneObjectPrefab = SceneObject;

class IFRIT_APIDECL Component : public Ifrit::Common::Utility::NonCopyable {
protected:
  ComponentIdentifier m_id;
  std::weak_ptr<SceneObject> m_parentObject;

public:
  Component(){}; // for deserializatioin
  Component(std::shared_ptr<SceneObject> parentObject);
  virtual ~Component() = default;
  virtual std::string serialize() = 0;
  virtual void deserialize() = 0;

  virtual void onPreRender() {}
  virtual void onPostRender() {}

  virtual void onStart() {}
  virtual void onUpdate() {}
  virtual void onEnd() {}

  inline void setName(const std::string &name) { m_id.m_name = name; }
  inline std::string getName() const { return m_id.m_name; }
  inline std::string getUUID() const { return m_id.m_uuid; }
  inline std::shared_ptr<SceneObject> getParent() const {
    return m_parentObject.lock();
  }

  virtual std::vector<AssetReference *> getAssetReferences() { return {}; }
  virtual void setAssetReferencedAttributes(
      const std::vector<std::shared_ptr<IAssetCompatible>> &out) {}

  IFRIT_STRUCT_SERIALIZE(m_id, m_parentObject);
};

struct TransformAttribute {
  ifloat3 m_position = ifloat3{0.0f, 0.0f, 0.0f};
  ifloat3 m_rotation = ifloat3{0.0f, 0.0f, 0.0f};
  ifloat3 m_scale = ifloat3{1.0f, 1.0f, 1.0f};

  IFRIT_STRUCT_SERIALIZE(m_position, m_rotation, m_scale);
};

class IFRIT_APIDECL Transform : public Component,
                                public AttributeOwner<TransformAttribute> {
private:
  using GPUUniformBuffer = Ifrit::GraphicsBackend::Rhi::RhiMultiBuffer;
  using GPUBindId = Ifrit::GraphicsBackend::Rhi::RhiBindlessIdRef;
  GPUUniformBuffer *m_gpuBuffer = nullptr;
  GPUUniformBuffer *m_gpuBufferLast = nullptr;
  std::shared_ptr<GPUBindId> m_gpuBindlessRef = nullptr;
  std::shared_ptr<GPUBindId> m_gpuBindlessRefLast = nullptr;

  TransformAttribute m_lastFrame;

  struct DirtyFlag {
    bool changed = true;
    bool lastChanged = true;
  } m_dirty;

public:
  Transform(){};
  Transform(std::shared_ptr<SceneObject> parent)
      : Component(parent), AttributeOwner<TransformAttribute>() {}
  std::string serialize() override { return serializeAttribute(); }
  void deserialize() override { deserializeAttribute(); }

  inline void onFrameCollecting() {
    if (m_dirty.changed) {
      m_lastFrame = m_attributes;
    }
    m_dirty.lastChanged = m_dirty.changed;
    m_dirty.changed = false;
  }

  // getters
  inline ifloat3 getPosition() const { return m_attributes.m_position; }
  inline ifloat3 getRotation() const { return m_attributes.m_rotation; }
  inline ifloat3 getScale() const { return m_attributes.m_scale; }

  // setters
  inline void setPosition(const ifloat3 &pos) {
    m_attributes.m_position = pos;
    m_dirty.changed = true;
  }
  inline void setRotation(const ifloat3 &rot) {
    m_attributes.m_rotation = rot;
    m_dirty.changed = true;
  }
  inline void setScale(const ifloat3 &scale) {
    m_attributes.m_scale = scale;
    m_dirty.changed = true;
  }

  inline void markUnchanged() { m_dirty.changed = false; }

  inline DirtyFlag getDirtyFlag() { return m_dirty; }

  float4x4 getModelToWorldMatrix();
  float4x4 getModelToWorldMatrixLast();
  inline ifloat3 getScaleLast() { return m_lastFrame.m_scale; }
  inline void setGPUResource(GPUUniformBuffer *buffer, GPUUniformBuffer *last,
                             std::shared_ptr<GPUBindId> &bindlessRef,
                             std::shared_ptr<GPUBindId> &bindlessRefLast) {
    m_gpuBuffer = buffer;
    m_gpuBufferLast = last;
    m_gpuBindlessRef = bindlessRef;
    m_gpuBindlessRefLast = bindlessRefLast;
  }
  inline void getGPUResource(GPUUniformBuffer *&buffer, GPUUniformBuffer *&last,
                             std::shared_ptr<GPUBindId> &bindlessRef,
                             std::shared_ptr<GPUBindId> &bindlessRefLast) {
    buffer = m_gpuBuffer;
    last = m_gpuBufferLast;
    bindlessRef = m_gpuBindlessRef;
    bindlessRefLast = m_gpuBindlessRefLast;
  }
  IFRIT_COMPONENT_SERIALIZE(m_attributes);
};

} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Transform);