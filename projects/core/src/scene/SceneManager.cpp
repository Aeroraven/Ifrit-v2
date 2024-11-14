#include "ifrit/core/scene/SceneManager.h"
#include "ifrit/common/util/TypingUtil.h"

using namespace Ifrit::Common::Utility;

namespace Ifrit::Core {
IFRIT_APIDECL void
SceneManager::collectPerframeData(PerFrameData &perframeData, Scene *scene,
                                  Camera *camera,
                                  GraphicsShaderPassType passType) {
  // Filling per frame data
  if (camera == nullptr) {
    camera = scene->getMainCamera();
  }
  if (camera == nullptr) {
    throw std::runtime_error("No camera found in scene");
  }
  perframeData.m_viewData.m_worldToView = camera->worldToCameraMatrix();
  perframeData.m_viewData.m_perspective = camera->projectionMatrix();
  perframeData.m_viewData.m_cameraAspect = camera->getAspect();
  auto cameraTransform = camera->getParent()->getComponent<Transform>();
  if (cameraTransform == nullptr) {
    throw std::runtime_error("Camera has no transform");
  }
  auto pos = cameraTransform->getPosition();
  perframeData.m_viewData.m_cameraPosition = ifloat4{pos.x, pos.y, pos.z, 1.0f};
  perframeData.m_viewData.m_cameraNear = camera->getNear();
  perframeData.m_viewData.m_cameraFar = camera->getFar();
  perframeData.m_viewData.m_cameraFovX = camera->getFov();
  perframeData.m_viewData.m_cameraFovY = camera->getFov();

  // For each mesh renderer, get the material, mesh, and transform
  perframeData.m_enabledEffects.clear();
  for (auto &effect : perframeData.m_shaderEffectData) {
    effect.m_materials.clear();
    effect.m_meshes.clear();
    effect.m_transforms.clear();
    effect.m_instances.clear();
  }

  std::vector<std::shared_ptr<Material>> materials;
  std::vector<std::shared_ptr<Mesh>> meshes;
  std::vector<std::shared_ptr<Transform>> transforms;
  std::vector<std::shared_ptr<MeshInstance>> instances;

  std::vector<SceneNode *> nodes;
  nodes.push_back(scene->getRootNode().get());
  while (!nodes.empty()) {
    auto node = nodes.back();
    nodes.pop_back();
    for (auto &child : node->getChildren()) {
      nodes.push_back(child.get());
    }
    for (auto &obj : node->getGameObjects()) {
      auto meshRenderer = obj->getComponent<MeshRenderer>();
      auto meshFilter = obj->getComponent<MeshFilter>();
      auto transform = obj->getComponent<Transform>();
      if (!meshRenderer || !meshFilter || !transform) {
        continue;
      }
      if (meshRenderer && meshFilter && transform) {
        materials.push_back(meshRenderer->getMaterial());
        meshes.push_back(meshFilter->getMesh());
        transforms.push_back(transform);
        instances.push_back(meshFilter->getMeshInstance());
      } else {
        throw std::runtime_error(
            "MeshRenderer, MeshFilter, or Transform not found");
      }
    }
  }

  // Groups meshes with the same shader effect
  for (size_t i = 0; i < materials.size(); i++) {
    auto &material = materials[i];
    auto &mesh = meshes[i];
    auto &transform = transforms[i];
    auto &instance = instances[i];

    ShaderEffect effect;
    effect.m_shaders = material->m_effectTemplates[passType].m_shaders;

    // TODO: Heavy copy operation, should be avoided
    effect.m_drawPasses = material->m_effectTemplates[passType].m_drawPasses;
    if (perframeData.m_shaderEffectMap.count(effect) == 0) {
      perframeData.m_shaderEffectMap[effect] =
          size_cast<uint32_t>(perframeData.m_shaderEffectData.size());
      perframeData.m_shaderEffectData.push_back(PerShaderEffectData{});
    }
    auto id = perframeData.m_shaderEffectMap[effect];
    auto &shaderEffectData = perframeData.m_shaderEffectData[id];
    perframeData.m_enabledEffects.insert(id);

    shaderEffectData.m_materials.push_back(material);
    shaderEffectData.m_meshes.push_back(mesh);
    shaderEffectData.m_transforms.push_back(transform);
    shaderEffectData.m_instances.push_back(instance);
  }
}
} // namespace Ifrit::Core