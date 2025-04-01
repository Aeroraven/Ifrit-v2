
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
#include "ifrit/runtime/application/Application.h"
#include "ifrit/runtime/assetmanager/Asset.h"
#include "ifrit/runtime/assetmanager/GLTFAsset.h"
#include "ifrit/runtime/assetmanager/ShaderAsset.h"
#include "ifrit/runtime/assetmanager/WaveFrontAsset.h"
#include "ifrit/runtime/base/Camera.h"
#include "ifrit/runtime/base/Component.h"
#include "ifrit/runtime/base/ActorBehavior.h"
#include "ifrit/runtime/base/Light.h"
#include "ifrit/runtime/base/Material.h"
#include "ifrit/runtime/base/Mesh.h"
#include "ifrit/runtime/base/Scene.h"
#include "ifrit/runtime/renderer/AyanamiRenderer.h"
#include "ifrit/runtime/renderer/SyaroRenderer.h"
#include "ifrit/runtime/scene/FrameCollector.h"
#include "ifrit/runtime/scene/SceneAssetManager.h"
#include "ifrit/runtime/scene/SceneManager.h"

#include "ifrit/runtime/renderer/ayanami/AyanamiMeshDF.h"
#include "ifrit/runtime/renderer/ayanami/AyanamiMeshMarker.h"