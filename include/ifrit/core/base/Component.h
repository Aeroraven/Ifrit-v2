#pragma once
#include "ifrit/common/math/VectorDefs.h"
#include "ifrit/common/serialization/MathTypeSerialization.h"
#include "ifrit/common/serialization/SerialInterface.h"
#include "ifrit/common/util/ApiConv.h"
#include "ifrit/common/util/TypingUtil.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#define IFRIT_COMPONENT_SERIALIZE(...) IFRIT_STRUCT_SERIALIZE(m_id,m_parentObject,__VA_ARGS__)
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

class IFRIT_APIDECL SceneObject
    : public Ifrit::Common::Utility::NonCopyable,
      public std::enable_shared_from_this<SceneObject> {
protected:
  ComponentIdentifier m_id;
  std::unordered_map<std::string, std::shared_ptr<Component>> m_components;

public:
  SceneObject();
  virtual ~SceneObject() = default;
  void initialize();

  template <class T> void addComponent(std::shared_ptr<T> component) {
    static_assert(std::is_base_of<Component, T>::value,
                  "T must be derived from Component");
    auto typeName = typeid(T).name();
    m_components[typeName] = component;
  }
  template <class T> std::shared_ptr<T> addComponent() {
    static_assert(std::is_base_of<Component, T>::value,
                  "T must be derived from Component");
    auto typeName = typeid(T).name();
    auto ret = std::make_shared<T>(shared_from_this());
    m_components[typeName] = ret;
    return ret;
  }

  template <class T> std::shared_ptr<T> getComponent() {
    static_assert(std::is_base_of<Component, T>::value,
                  "T must be derived from Component");
    auto typeName = typeid(T).name();
    auto it = m_components.find(typeName);
    if (it != m_components.end()) {
      return std::dynamic_pointer_cast<T>(it->second);
    }
    return nullptr;
  }
  IFRIT_STRUCT_SERIALIZE(m_id, m_components);
};

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
public:
  Transform(){};
  Transform(std::shared_ptr<SceneObject> parent)
      : Component(parent), AttributeOwner<TransformAttribute>() {}
  std::string serialize() override { return serializeAttribute(); }
  void deserialize() override { deserializeAttribute(); }

  // getters
  inline ifloat3 getPosition() const { return m_attributes.m_position; }
  inline ifloat3 getRotation() const { return m_attributes.m_rotation; }
  inline ifloat3 getScale() const { return m_attributes.m_scale; }

  // setters
  inline void setPosition(const ifloat3 &pos) { m_attributes.m_position = pos; }
  inline void setRotation(const ifloat3 &rot) { m_attributes.m_rotation = rot; }
  inline void setScale(const ifloat3 &scale) { m_attributes.m_scale = scale; }

  IFRIT_COMPONENT_SERIALIZE(m_attributes);
};

} // namespace Ifrit::Core

IFRIT_COMPONENT_REGISTER(Ifrit::Core::Transform);