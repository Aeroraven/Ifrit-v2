
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
#define IFRIT_BINDLESS_BINDING_SHADER_READ_STORAGE 1    // Ignored in vulkan
#define IFRIT_BINDLESS_BINDING_STORAGE 2
#define IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER 3
#define IFRIT_BINDLESS_BINDING_STORAGE_IMAGE 4
#define IFRIT_BINDLESS_BINDING_SAMPLED_IMAGE 5
#define IFRIT_BINDLESS_BINDING_SAMPLER 6
#define IFRIT_BINDLESS_SET_ID 0

#define _ifrit_bindlessNaming(name) u##name##_bindless
#define _ifrit_bindlessType(name) u##name##_bindless_type

#ifdef IF_VERTEX_SHADER
    #define IFSHADER_LEGACY_SSBO_ACCESS readonly
#else
    #define IFSHADER_LEGACY_SSBO_ACCESS
#endif

#define RegisterUniform(name, type) layout(binding = IFRIT_BINDLESS_BINDING_UNIFORM, set = IFRIT_BINDLESS_SET_ID) \
    uniform _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]

#define RegisterStorage(name, type) layout(binding = IFRIT_BINDLESS_BINDING_STORAGE, set = IFRIT_BINDLESS_SET_ID) \
    IFSHADER_LEGACY_SSBO_ACCESS buffer _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]

#define RegisterStorage140(name, type) layout(std140,binding = IFRIT_BINDLESS_BINDING_STORAGE, set = IFRIT_BINDLESS_SET_ID) \
    IFSHADER_LEGACY_SSBO_ACCESS buffer _ifrit_bindlessType(name) type _ifrit_bindlessNaming(name)[]


// combined image samplers
#define IFRIT_BINDLESS_SAMPLER2D_NAME _ifrit_bindlessNaming(cbsampler_2d)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform sampler2D IFRIT_BINDLESS_SAMPLER2D_NAME[];

#define IFRIT_BINDLESS_SAMPLER2DU_NAME _ifrit_bindlessNaming(cbsampler_2du)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform usampler2D IFRIT_BINDLESS_SAMPLER2DU_NAME[];

#define IFRIT_BINDLESS_SAMPLER3D_NAME _ifrit_bindlessNaming(cbsampler_3d)
layout(binding = IFRIT_BINDLESS_BINDING_COMBINED_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform sampler3D IFRIT_BINDLESS_SAMPLER3D_NAME[];


// UAV textures

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

#ifdef IFRIT_USE_INT64_IMAGE
#define IFRIT_UAV_IMAGE2D_R64UI_NAME _ifrit_bindlessNaming(uav_image2d_r64ui)
layout(binding = IFRIT_BINDLESS_BINDING_STORAGE_IMAGE, set = IFRIT_BINDLESS_SET_ID, r64ui) uniform u64image2D IFRIT_UAV_IMAGE2D_R64UI_NAME[];
#endif

// SRV textures
#define IFRIT_SRV_TEXTURE2D_NAME _ifrit_bindlessNaming(srv_texture_2d)
layout(binding = IFRIT_BINDLESS_BINDING_SAMPLED_IMAGE, set = IFRIT_BINDLESS_SET_ID) uniform texture2D IFRIT_SRV_TEXTURE2D_NAME[];

#define IFRIT_SRV_TEXTURE2DU_NAME _ifrit_bindlessNaming(srv_texture_2du)
layout(binding = IFRIT_BINDLESS_BINDING_SAMPLED_IMAGE, set = IFRIT_BINDLESS_SET_ID) uniform utexture2D IFRIT_SRV_TEXTURE2DU_NAME[];

#define IFRIT_SRV_TEXTURE3D_NAME _ifrit_bindlessNaming(srv_texture_3d)
layout(binding = IFRIT_BINDLESS_BINDING_SAMPLED_IMAGE, set = IFRIT_BINDLESS_SET_ID) uniform texture3D IFRIT_SRV_TEXTURE3D_NAME[];

// Samplers
#define IFRIT_BINDLESS_INDEP_SAMPLER_NAME _ifrit_bindlessNaming(indep_sampler)
layout(binding = IFRIT_BINDLESS_BINDING_SAMPLER, set = IFRIT_BINDLESS_SET_ID) uniform sampler IFRIT_BINDLESS_INDEP_SAMPLER_NAME[];

#define GetResource(name,id) _ifrit_bindlessNaming(name)[(id)]

// Update 250503: to make the impl compatible with other rhi impl, combined samplers are dropped forcely
// together with the rhi-side apis.
// 
// #define GetSampler2D(id) IFRIT_BINDLESS_SAMPLER2D_NAME[(id)]
// #define GetSampler3D(id) IFRIT_BINDLESS_SAMPLER3D_NAME[(id)]
// #define GetSampler2DU(id) IFRIT_BINDLESS_SAMPLER2DU_NAME[(id)]

#define GetUAVImage2DR64UI(id) IFRIT_UAV_IMAGE2D_R64UI_NAME[(id)]
#define GetUAVImage2DR32F(id) IFRIT_UAV_IMAGE2D_R32F_NAME[(id)]
#define GetUAVImage2DR32UI(id) IFRIT_UAV_IMAGE2D_R32UI_NAME[(id)]
#define GetUAVImage2DRGBA32F(id) IFRIT_UAV_IMAGE2D_RGBA32F_NAME[(id)]
#define GetUAVImage3DRGBA32F(id) IFRIT_UAV_IMAGE3D_RGBA32F_NAME[(id)]
#define GetUAVImage3DR32F(id) IFRIT_UAV_IMAGE3D_R32F_NAME[(id)]
#define GetUAVImage2DRGBA8(id) IFRIT_UAV_IMAGE2D_RGBA8_NAME[(id)]

#define SampleTexture2D(texId, samplerId, uv) texture(sampler2D(IFRIT_SRV_TEXTURE2D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), uv)
#define SampleTexture2DOffset(texId, samplerId, uv, offset) textureOffset(sampler2D(IFRIT_SRV_TEXTURE2D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), uv, offset)
#define SampleTexture2DSize(texId, samplerId) textureSize(sampler2D(IFRIT_SRV_TEXTURE2D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), 0)

#define SampleTexture2DUint(texId, samplerId, uv) texture(usampler2D(IFRIT_SRV_TEXTURE2DU_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), uv)

#define SampleTexture3D(texId, samplerId, uv) texture(sampler3D(IFRIT_SRV_TEXTURE3D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), uv)

#define SampleTexture2DLoad(texId, samplerId, uv) texelFetch(sampler2D(IFRIT_SRV_TEXTURE2D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), uv,0)
#define SampleTexture2DLoadUint(texId, samplerId, uv) texelFetch(usampler2D(IFRIT_SRV_TEXTURE2DU_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), uv,0)
#define SampleTexture3DLoad(texId, samplerId, uv) texelFetch(sampler3D(IFRIT_SRV_TEXTURE3D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId]), uv,0)

#define CombineSampler2D(texId, samplerId) sampler2D(IFRIT_SRV_TEXTURE2D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId])
#define CombineSampler3D(texId, samplerId) sampler3D(IFRIT_SRV_TEXTURE3D_NAME[texId], IFRIT_BINDLESS_INDEP_SAMPLER_NAME[samplerId])

RegisterStorage(bIfritInternal_VerticesPos,{
    vec4 position[];
});

RegisterStorage(bIfritInternal_NormalPos,{
    vec4 normal[];
});

RegisterStorage(bIfritInternal_TangentPos,{
    vec4 tangent[];
});

RegisterStorage(bIfritInternal_VerticesUV,{
    vec2 uv[];
});

#define ReadVertexPosition(objId,vertexId) GetResource(bIfritInternal_VerticesPos,objId).position[vertexId]
#define ReadVertexNormal(objId,vertexId) GetResource(bIfritInternal_NormalPos,objId).normal[vertexId]
#define ReadVertexTangent(objId,vertexId) GetResource(bIfritInternal_TangentPos,objId).tangent[vertexId]
#define ReadVertexUV(objId,vertexId) GetResource(bIfritInternal_VerticesUV,objId).uv[vertexId]