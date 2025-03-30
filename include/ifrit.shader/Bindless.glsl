
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


#extension GL_EXT_nonuniform_qualifier : enable

#define IFRIT_BINDLESS_BINDING_UNIFORM 0
#define IFRIT_BINDLESS_BINDING_STORAGE 1
#define IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER 2
#define IFRIT_BINDLESS_BINDING_STORAGE_IMAGE 3
#define IFRIT_BINDLESS_SET_ID 0

#define _ifrit_bindlessNaming(name) u##name##_bindless
#define _ifrit_bindlessType(name) u##name##_bindless_type

#define RegisterUniform(name, type) layout(binding = IFRIT_BINDLESS_BINDING_UNIFORM, set = IFRIT_BINDLESS_SET_ID) \
    uniform _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]

#define RegisterStorage(name, type) layout(binding = IFRIT_BINDLESS_BINDING_STORAGE, set = IFRIT_BINDLESS_SET_ID) \
    buffer _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]

#define RegisterStorage140(name, type) layout(std140,binding = IFRIT_BINDLESS_BINDING_STORAGE, set = IFRIT_BINDLESS_SET_ID) \
    buffer _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]

#define IFRIT_BINDLESS_SAMPLER2D_NAME _ifrit_bindlessNaming(cbsampler_2d)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform sampler2D IFRIT_BINDLESS_SAMPLER2D_NAME[];

#define IFRIT_BINDLESS_SAMPLER2DU_NAME _ifrit_bindlessNaming(cbsampler_2du)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform usampler2D IFRIT_BINDLESS_SAMPLER2DU_NAME[];

#define IFRIT_BINDLESS_SAMPLER3D_NAME _ifrit_bindlessNaming(cbsampler_3d)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform sampler3D IFRIT_BINDLESS_SAMPLER3D_NAME[];

#define IFRIT_UAV_IMAGE2D_R32UI_NAME _ifrit_bindlessNaming(uav_image2d_r32ui)
layout(binding = IFRIT_BINDLESS_BINDING_STORAGE_IMAGE, set = IFRIT_BINDLESS_SET_ID, r32ui) uniform uimage2D IFRIT_UAV_IMAGE2D_R32UI_NAME[];

#define IFRIT_UAV_IMAGE2D_R32F_NAME _ifrit_bindlessNaming(uav_image2d_r32f) 
layout(binding = IFRIT_BINDLESS_BINDING_STORAGE_IMAGE, set = IFRIT_BINDLESS_SET_ID, r32f) uniform image2D IFRIT_UAV_IMAGE2D_R32F_NAME[];

#define IFRIT_UAV_IMAGE2D_RGBA32F_NAME _ifrit_bindlessNaming(uav_image2d_rgba32f)
layout(binding = IFRIT_BINDLESS_BINDING_STORAGE_IMAGE, set = IFRIT_BINDLESS_SET_ID, rgba32f) uniform image2D IFRIT_UAV_IMAGE2D_RGBA32F_NAME[];

#define IFRIT_UAV_IMAGE2D_RGBA8_NAME _ifrit_bindlessNaming(uav_image2d_rgba8)
layout(binding = IFRIT_BINDLESS_BINDING_STORAGE_IMAGE, set = IFRIT_BINDLESS_SET_ID, rgba8) uniform image2D IFRIT_UAV_IMAGE2D_RGBA8_NAME[];

#define IFRIT_UAV_IMAGE3D_RGBA32F_NAME _ifrit_bindlessNaming(uav_image3d_rgba32f)
layout(binding = IFRIT_BINDLESS_BINDING_STORAGE_IMAGE, set = IFRIT_BINDLESS_SET_ID, rgba32f) uniform image3D IFRIT_UAV_IMAGE3D_RGBA32F_NAME[];

#define IFRIT_UAV_IMAGE3D_R32F_NAME _ifrit_bindlessNaming(uav_image3d_r32f)
layout(binding = IFRIT_BINDLESS_BINDING_STORAGE_IMAGE, set = IFRIT_BINDLESS_SET_ID, r32f) uniform image3D IFRIT_UAV_IMAGE3D_R32F_NAME[];


#define GetResource(name,id) _ifrit_bindlessNaming(name)[(id)]
#define GetSampler2D(id) IFRIT_BINDLESS_SAMPLER2D_NAME[(id)]
#define GetSampler3D(id) IFRIT_BINDLESS_SAMPLER3D_NAME[(id)]
#define GetSampler2DU(id) IFRIT_BINDLESS_SAMPLER2DU_NAME[(id)]

#define GetUAVImage2DR32F(id) IFRIT_UAV_IMAGE2D_R32F_NAME[(id)]
#define GetUAVImage2DR32UI(id) IFRIT_UAV_IMAGE2D_R32UI_NAME[(id)]
#define GetUAVImage2DRGBA32F(id) IFRIT_UAV_IMAGE2D_RGBA32F_NAME[(id)]
#define GetUAVImage3DRGBA32F(id) IFRIT_UAV_IMAGE3D_RGBA32F_NAME[(id)]
#define GetUAVImage3DR32F(id) IFRIT_UAV_IMAGE3D_R32F_NAME[(id)]
#define GetUAVImage2DRGBA8(id) IFRIT_UAV_IMAGE2D_RGBA8_NAME[(id)]

RegisterStorage(bIfritInternal_VerticesPos,{
    vec4 position[];
});

RegisterStorage(bIfritInternal_VerticesUV,{
    vec2 uv[];
});

#define ReadVertexPosition(objId,vertexId) GetResource(bIfritInternal_VerticesPos,objId).position[vertexId]
#define ReadVertexUV(objId,vertexId) GetResource(bIfritInternal_VerticesUV,objId).uv[vertexId]